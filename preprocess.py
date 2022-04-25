import pandas as pd
import csv
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import matplotlib.dates as mdates
# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
font1 = {'family': 'Microsoft YaHei',
         'weight': 'normal',
         'size': 13}
plt.style.use(['science','no-latex'])
def get_sent_recv(s):
    # 1(Sent) + 1(Recv
    position_jiahao = s.find('+')
    s1 = s[0:position_jiahao-7]
    s2 = s[position_jiahao+2:]
    s2_left_pos = s2.find('(')
    s2 = s2[0:s2_left_pos]
    # print("s = {}".format(s))
    # print("s1 = {}".format(s1))
    # print("s2 = {}".format(s2))
    s2_left_position = s2.find('(')
    sent = eval(s1)
    recv = eval(s2)
    return sent, recv

exchanges = ["binance", "coinbase", "huobi", "kraken", "kucoin"]

def save_features():
    for i in range(len(exchanges)):
        print("i = {}".format(i))
        exchange = exchanges[i]
        file_address = open("./exchange/" + exchange + ".csv")
        df_address = pd.read_csv(file_address)
        addresses = df_address['address']

        # calculate every exchange's information
        # plotdata1
        ether_account_balance = {}
        usd_eth = {}
        historic_usd_val = {}
        transaction_count_total = {}
        transaction_count_sent = {}
        transaction_count_recv = {}
        eth_fee_spent = {}
        eth_fee_used = {}
        # plotdata2
        token_transfers = {}
        outbound_transfers = {}
        inbound_transfers = {}
        unique_address_sent = {}
        unique_address_recv = {}
        token_contracts_count = {}
        # plotdata3
        ethereum_transactions = {}
        unique_outgoing_address = {}
        unique_incoming_address = {}
        # plotdata6
        ether_sent_out = {}
        ether_recv_in = {}

        for k in range(len(addresses)):
            # print("k = {}".format(k))
            address = addresses[k]
            data1 = json.load(open("./exchange/" + exchange + "/" + address + "/plotdata1.json"))
            data2 = json.load(open("./exchange/" + exchange + "/" + address + "/plotdata2.json"))
            data3 = json.load(open("./exchange/" + exchange + "/" + address + "/plotdata3.json"))
            data6 = json.load(open("./exchange/" + exchange + "/" + address + "/plotdata6.json"))
            # b1 = [d[0] for d in data1]
            for j in range(len(data1)):
                # print("j = {}".format(j))
                time_stamp = data1[j][0]
                time_stamp = int(time_stamp / 1000)
                # print(time_stamp)
                if time_stamp in ether_account_balance:
                    ether_account_balance[time_stamp] = ether_account_balance[time_stamp] + data1[j][1]
                    usd_eth[time_stamp] = data1[j][2]
                    historic_usd_val[time_stamp] = historic_usd_val[time_stamp] + data1[j][3]
                    transaction_count_total[time_stamp] = transaction_count_total[time_stamp] + data1[j][5]
                    temp_sent, temp_recv = get_sent_recv(data1[j][6])
                    transaction_count_sent[time_stamp] = transaction_count_sent[time_stamp] + temp_sent
                    transaction_count_recv[time_stamp] = transaction_count_recv[time_stamp] + temp_recv
                    eth_fee_spent[time_stamp] = eth_fee_spent[time_stamp] + data1[j][7]
                    eth_fee_used[time_stamp] = eth_fee_used[time_stamp] + data1[j][9]
                else:
                    ether_account_balance[time_stamp] = data1[j][1]
                    usd_eth[time_stamp] = data1[j][2]
                    historic_usd_val[time_stamp] = data1[j][3]
                    transaction_count_total[time_stamp] = data1[j][5]
                    temp_sent, temp_recv = get_sent_recv(data1[j][6])
                    transaction_count_sent[time_stamp] = temp_sent
                    transaction_count_recv[time_stamp] = temp_recv
                    eth_fee_spent[time_stamp] = data1[j][7]
                    eth_fee_used[time_stamp] = data1[j][9]
            for j in range(len(data2)):
                time_stamp = data2[j][0]
                time_stamp = int(time_stamp / 1000)
                if time_stamp in token_transfers:
                    token_transfers[time_stamp] = token_transfers[time_stamp] + data2[j][1]
                    outbound_transfers[time_stamp] = data2[j][2] + outbound_transfers[time_stamp]
                    inbound_transfers[time_stamp] = data2[j][3] + inbound_transfers[time_stamp]
                    unique_address_sent[time_stamp] = data2[j][4] + unique_address_sent[time_stamp]
                    unique_address_recv[time_stamp] = data2[j][5] + unique_address_recv[time_stamp]
                    # print("i = {}, k = {}, j = {}".format(i, k, j))
                    # print("address = {}".format(address))
                    # print("token_contracts_count[time_stamp]")
                    # print(token_contracts_count[time_stamp])
                    # print("data2[j][6]")
                    # print(data2[j][6])
                    token_contracts_count[time_stamp] = data2[j][6] + token_contracts_count[time_stamp]
                else:
                    token_transfers[time_stamp] = data2[j][1]
                    outbound_transfers[time_stamp] = data2[j][2]
                    inbound_transfers[time_stamp] = data2[j][3]
                    unique_address_sent[time_stamp] = data2[j][4]
                    unique_address_recv[time_stamp] = data2[j][5]
                    token_contracts_count[time_stamp] = data2[j][6]
            for j in range(len(data3)):
                time_stamp = data3[j][0]
                time_stamp = int(time_stamp / 1000)
                if time_stamp in ethereum_transactions:
                    ethereum_transactions[time_stamp] = data3[j][1] + ethereum_transactions[time_stamp]
                    unique_outgoing_address[time_stamp] = data3[j][4] + unique_outgoing_address[time_stamp]
                    unique_incoming_address[time_stamp] = data3[j][5] + unique_incoming_address[time_stamp]
                else:
                    ethereum_transactions[time_stamp] = data3[j][1]
                    unique_outgoing_address[time_stamp] = data3[j][4]
                    unique_incoming_address[time_stamp] = data3[j][5]
            for j in range(len(data6)):
                time_stamp = data6[j][0]
                time_stamp = int(time_stamp / 1000)
                if time_stamp in ether_sent_out:
                    ether_sent_out[time_stamp] = data6[j][1] + ether_sent_out[time_stamp]
                    ether_recv_in[time_stamp] = data6[j][2] + ether_recv_in[time_stamp]
                else:
                    ether_sent_out[time_stamp] = data6[j][1]
                    ether_recv_in[time_stamp] = data6[j][2]

        print("plotdata1")
        print(len(ether_account_balance))
        print(len(usd_eth))
        print(len(historic_usd_val))
        print(len(transaction_count_total))
        print(len(transaction_count_sent))
        print(len(transaction_count_recv))
        print(len(eth_fee_spent))
        print(len(eth_fee_used))
        print("plotdata2")
        print(len(token_transfers))
        print(len(outbound_transfers))
        print(len(inbound_transfers))
        print(len(unique_address_sent))
        print(len(unique_address_recv))
        print(len(token_contracts_count))
        print("plotdata3")
        print(len(ethereum_transactions))
        print(len(unique_outgoing_address))
        print(len(unique_incoming_address))
        print("plotdata6")
        print(len(ether_sent_out))
        print(len(ether_recv_in))

        np.save('./exchange/' + exchange + '/ether_account_balance.npy', ether_account_balance)
        np.save('./exchange/' + exchange + '/usd_eth.npy', usd_eth)
        np.save('./exchange/' + exchange + '/historic_usd_val.npy', historic_usd_val)
        np.save('./exchange/' + exchange + '/transaction_count_total.npy', transaction_count_total)
        np.save('./exchange/' + exchange + '/transaction_count_sent.npy', transaction_count_sent)
        np.save('./exchange/' + exchange + '/transaction_count_recv.npy', transaction_count_recv)
        np.save('./exchange/' + exchange + '/eth_fee_spent.npy', eth_fee_spent)
        np.save('./exchange/' + exchange + '/eth_fee_used.npy', eth_fee_used)

        np.save('./exchange/' + exchange + '/token_transfers.npy', token_transfers)
        np.save('./exchange/' + exchange + '/outbound_transfers.npy', outbound_transfers)
        np.save('./exchange/' + exchange + '/inbound_transfers.npy', inbound_transfers)
        np.save('./exchange/' + exchange + '/unique_address_sent.npy', unique_address_sent)
        np.save('./exchange/' + exchange + '/unique_address_recv.npy', unique_address_recv)
        np.save('./exchange/' + exchange + '/token_contracts_count.npy', token_contracts_count)

        np.save('./exchange/' + exchange + '/ethereum_transactions.npy', ethereum_transactions)
        np.save('./exchange/' + exchange + '/unique_outgoing_address.npy', unique_outgoing_address)
        np.save('./exchange/' + exchange + '/unique_incoming_address.npy', unique_incoming_address)

        np.save('./exchange/' + exchange + '/ether_sent_out.npy', ether_sent_out)
        np.save('./exchange/' + exchange + '/ether_recv_in.npy', ether_recv_in)

def min_max_scaler(li):
    max_ = max(li)
    min_ = min(li)
    li = [(i - min_)/(max_ - min_) for i in li]
    return li
def load_features():
    for i in range(len(exchanges)):
        exchange = exchanges[i]
        print('i = {}'.format(i))

        ether_account_balance = np.load('./exchange/' + exchange + '/ether_account_balance.npy', allow_pickle=True).item()
        usd_eth = np.load('./exchange/' + exchange + '/usd_eth.npy', allow_pickle=True).item()
        historic_usd_val = np.load('./exchange/' + exchange + '/historic_usd_val.npy', allow_pickle=True).item()
        transaction_count_total = np.load('./exchange/' + exchange + '/transaction_count_total.npy', allow_pickle=True).item()
        transaction_count_sent = np.load('./exchange/' + exchange + '/transaction_count_sent.npy', allow_pickle=True).item()
        transaction_count_recv = np.load('./exchange/' + exchange + '/transaction_count_recv.npy', allow_pickle=True).item()
        eth_fee_spent = np.load('./exchange/' + exchange + '/eth_fee_spent.npy', allow_pickle=True).item()
        eth_fee_used = np.load('./exchange/' + exchange + '/eth_fee_used.npy', allow_pickle=True).item()
        time_stamp_list_1 = []
        ether_account_balance_list = []
        usd_eth_list = []
        historic_usd_val_list = []
        transaction_count_total_list = []
        transaction_count_sent_list = []
        transaction_count_recv_list = []
        eth_fee_spent_list = []
        eth_fee_used_list = []
        token_transfers = np.load('./exchange/' + exchange + '/token_transfers.npy', allow_pickle=True).item()
        outbound_transfers = np.load('./exchange/' + exchange + '/outbound_transfers.npy', allow_pickle=True).item()
        inbound_transfers = np.load('./exchange/' + exchange + '/inbound_transfers.npy', allow_pickle=True).item()
        unique_address_sent = np.load('./exchange/' + exchange + '/unique_address_sent.npy', allow_pickle=True).item()
        unique_address_recv = np.load('./exchange/' + exchange + '/unique_address_recv.npy', allow_pickle=True).item()
        token_contracts_count = np.load('./exchange/' + exchange + '/token_contracts_count.npy',
                                        allow_pickle=True).item()
        time_stamp_list_2 = []
        token_transfers_list = []
        outbound_transfers_list = []
        inbound_transfers_list = []
        unique_address_sent_list = []
        unique_address_recv_list = []
        token_contracts_count_list = []
        ethereum_transactions = np.load('./exchange/' + exchange + '/ethereum_transactions.npy',
                                        allow_pickle=True).item()
        unique_outgoing_address = np.load('./exchange/' + exchange + '/unique_outgoing_address.npy',
                                          allow_pickle=True).item()
        unique_incoming_address = np.load('./exchange/' + exchange + '/unique_incoming_address.npy',
                                          allow_pickle=True).item()
        time_stamp_list_3 = []
        ethereum_transactions_list = []
        unique_outgoing_address_list = []
        unique_incoming_address_list = []
        ether_sent_out = np.load('./exchange/' + exchange + '/ether_sent_out.npy', allow_pickle=True).item()
        ether_recv_in = np.load('./exchange/' + exchange + '/ether_recv_in.npy', allow_pickle=True).item()
        time_stamp_list_4 = []
        ether_sent_out_list = []
        ether_recv_in_list = []
        ether_usd_sent_out_list = []
        ether_usd_recv_in_list = []
        ether_usd_total_list = []
        transaction_amount_usd_origin = []

        # if exchange == 'okex':
        #     for key in sorted(ether_account_balance):
        #         time_stamp_list_1.append(key)
        #     minTime = min(time_stamp_list_1)
        #     maxTime = max(time_stamp_list_1)
        #     print("minTime = {}".format(minTime))
        #     print("maxTime = {}".format(maxTime))
        #     # 对前一天和后一天中间的值做均值或取零
        #     print("前 ether_sent_out = {}".format(len(ether_sent_out)))
        #     print("应有{}天".format((maxTime - minTime) / 86400 + 1))
        #     for it in range(len(time_stamp_list_1)):
        #         if it != 0 and time_stamp_list_1[it] - time_stamp_list_1[it - 1] != 86400:
        #             print("okex")
        #             print("it = {}".format(it-1))
        #             print(time_stamp_list_1[it-1])
        #             timeArray = time.localtime(time_stamp_list_1[it-1])
        #             otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        #             print(otherStyleTime)
        #             print("it = {}".format(it))
        #             print(time_stamp_list_1[it])
        #             timeArray = time.localtime(time_stamp_list_1[it])
        #             otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        #             print(otherStyleTime)
        #             ether_account_balance[time_stamp_list_1[it] - 86400] = (ether_account_balance[
        #                                                                         time_stamp_list_1[it]] +
        #                                                                     ether_account_balance[
        #                                                                         time_stamp_list_1[it - 1]]) / 2
        #             usd_eth[time_stamp_list_1[it] - 86400] = (usd_eth[time_stamp_list_1[it]] + usd_eth[
        #                 time_stamp_list_1[it - 1]]) / 2
        #             print(usd_eth[time_stamp_list_1[it] - 86400])
        #             historic_usd_val[time_stamp_list_1[it] - 86400] = (historic_usd_val[time_stamp_list_1[it]] +
        #                                                                historic_usd_val[time_stamp_list_1[it - 1]]) / 2
        #             transaction_count_total[time_stamp_list_1[it] - 86400] = 0
        #             transaction_count_sent[time_stamp_list_1[it] - 86400] = 0
        #             transaction_count_recv[time_stamp_list_1[it] - 86400] = 0
        #             eth_fee_spent[time_stamp_list_1[it] - 86400] = 0
        #             eth_fee_used[time_stamp_list_1[it] - 86400] = 0
        #             token_transfers[time_stamp_list_1[it] - 86400] = 0
        #             outbound_transfers[time_stamp_list_1[it] - 86400] = 0
        #             inbound_transfers[time_stamp_list_1[it] - 86400] = 0
        #             unique_address_sent[time_stamp_list_1[it] - 86400] = 0
        #             unique_address_recv[time_stamp_list_1[it] - 86400] = 0
        #             token_contracts_count[time_stamp_list_1[it] - 86400] = 0
        #             ethereum_transactions[time_stamp_list_1[it] - 86400] = 0
        #             unique_outgoing_address[time_stamp_list_1[it] - 86400] = 0
        #             unique_incoming_address[time_stamp_list_1[it] - 86400] = 0
        #             ether_sent_out[time_stamp_list_1[it] - 86400] = 0
        #             ether_recv_in[time_stamp_list_1[it] - 86400] = 0
        #
        #     print("后 ether_sent_out = {}".format(len(ether_sent_out)))
        #     time_stamp_list_1 = []  # 记得置空

        if exchange == 'kucoin':
            for key in sorted(ether_account_balance):
                time_stamp_list_1.append(key)
            minTime = min(time_stamp_list_1)
            maxTime = max(time_stamp_list_1)
            print("minTime = {}".format(minTime))
            print("maxTime = {}".format(maxTime))
            # 对前一天和后一天中间的值做均值或取零
            print("前 ether_sent_out = {}".format(len(ether_sent_out)))
            print("应有{}天".format((maxTime-minTime)/86400+1))
            for it in range(len(time_stamp_list_1)):
                if it != 0 and time_stamp_list_1[it] - time_stamp_list_1[it - 1] != 86400:
                    # print("it = {}".format(it-1))
                    # print(time_stamp_list_1[it-1])
                    # timeArray = time.localtime(time_stamp_list_1[it-1])
                    # otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
                    # print(otherStyleTime)
                    # print("it = {}".format(it))
                    # print(time_stamp_list_1[it])
                    # timeArray = time.localtime(time_stamp_list_1[it])
                    # otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
                    # print(otherStyleTime)
                    ether_account_balance[time_stamp_list_1[it] - 86400] = (ether_account_balance[
                                                                                time_stamp_list_1[it]] +
                                                                            ether_account_balance[
                                                                                time_stamp_list_1[it - 1]]) / 2
                    usd_eth[time_stamp_list_1[it] - 86400] = (usd_eth[time_stamp_list_1[it]] + usd_eth[
                        time_stamp_list_1[it - 1]]) / 2
                    print(usd_eth[time_stamp_list_1[it] - 86400])
                    historic_usd_val[time_stamp_list_1[it] - 86400] = (historic_usd_val[time_stamp_list_1[it]] +
                                                                       historic_usd_val[time_stamp_list_1[it - 1]]) / 2
                    transaction_count_total[time_stamp_list_1[it] - 86400] = 0
                    transaction_count_sent[time_stamp_list_1[it] - 86400] = 0
                    transaction_count_recv[time_stamp_list_1[it] - 86400] = 0
                    eth_fee_spent[time_stamp_list_1[it] - 86400] = 0
                    eth_fee_used[time_stamp_list_1[it] - 86400] = 0
                    token_transfers[time_stamp_list_1[it] - 86400] = 0
                    outbound_transfers[time_stamp_list_1[it] - 86400] = 0
                    inbound_transfers[time_stamp_list_1[it] - 86400] = 0
                    unique_address_sent[time_stamp_list_1[it] - 86400] = 0
                    unique_address_recv[time_stamp_list_1[it] - 86400] = 0
                    token_contracts_count[time_stamp_list_1[it] - 86400] = 0
                    ethereum_transactions[time_stamp_list_1[it] - 86400] = 0
                    unique_outgoing_address[time_stamp_list_1[it] - 86400] = 0
                    unique_incoming_address[time_stamp_list_1[it] - 86400] = 0
                    ether_sent_out[time_stamp_list_1[it] - 86400] = 0
                    ether_recv_in[time_stamp_list_1[it] - 86400] = 0

            print("后 ether_sent_out = {}".format(len(ether_sent_out)))
            time_stamp_list_1 = []  # 记得置空
        for key in sorted(ether_account_balance):
            time_stamp_list_1.append(key)
            ether_account_balance_list.append(ether_account_balance[key])
            usd_eth_list.append(usd_eth[key])
            historic_usd_val_list.append(historic_usd_val[key])
            transaction_count_total_list.append(transaction_count_total[key])
            transaction_count_sent_list.append(transaction_count_sent[key])
            transaction_count_recv_list.append(transaction_count_recv[key])
            eth_fee_spent_list.append(eth_fee_spent[key])
            eth_fee_used_list.append(eth_fee_used[key])

        print("exchange = {}".format(exchange))
        print("plotdata1")
        minTime = min(time_stamp_list_1)
        maxTime = max(time_stamp_list_1)
        print("最大时间-最小时间有{}天".format((maxTime-minTime)/86400 + 1))
        print("list包含{}项条目".format(len(time_stamp_list_1)))

        # 插值，补0
        for key in sorted(ether_account_balance):
            if key not in token_transfers:
               token_transfers[key] = 0
               outbound_transfers[key] = 0
               inbound_transfers[key] = 0
               unique_address_sent[key] = 0
               unique_address_recv[key] = 0
               token_contracts_count[key] = 0
        for key in sorted(token_transfers):
            time_stamp_list_2.append(key)
            token_transfers_list.append(token_transfers[key])
            outbound_transfers_list.append(outbound_transfers[key])
            inbound_transfers_list.append(inbound_transfers[key])
            unique_address_sent_list.append(unique_address_sent[key])
            unique_address_recv_list.append(unique_address_recv[key])
            token_contracts_count_list.append(token_contracts_count[key])
        print("plotdata2")
        minTime = min(time_stamp_list_2)
        maxTime = max(time_stamp_list_2)
        print("最大时间-最小时间有{}天".format((maxTime - minTime) / 86400 + 1))
        print("list包含{}项条目".format(len(time_stamp_list_2)))


        for key in sorted(ethereum_transactions):
            time_stamp_list_3.append(key)
            ethereum_transactions_list.append(ethereum_transactions[key])
            unique_outgoing_address_list.append(unique_outgoing_address[key])
            unique_incoming_address_list.append(unique_incoming_address[key])
        print("plotdata3")
        minTime = min(time_stamp_list_3)
        maxTime = max(time_stamp_list_3)
        print("最大时间-最小时间有{}天".format((maxTime - minTime) / 86400 + 1))
        print("list包含{}项条目".format(len(ethereum_transactions_list)))

        print("ether_sent_out = {}".format(len(ether_sent_out)))
        for key in sorted(ether_sent_out):
            time_stamp_list_4.append(key)
            ether_sent_out_list.append(ether_sent_out[key])
            ether_recv_in_list.append(ether_recv_in[key])
        print("plotdata6")
        minTime = min(time_stamp_list_4)
        maxTime = max(time_stamp_list_4)
        print("最大时间-最小时间有{}天".format((maxTime - minTime) / 86400 + 1))
        print("list包含{}项条目".format(len(ether_sent_out_list)))

        # 减去120天
        if True:
            if exchange == "huobi" or exchange == "kraken":
                time_stamp_list_1 = time_stamp_list_1[0:len(time_stamp_list_1) - 100]
                time_stamp_list_2 = time_stamp_list_2[0:len(time_stamp_list_2) - 100]
                time_stamp_list_3 = time_stamp_list_3[0:len(time_stamp_list_3) - 100]
                time_stamp_list_4 = time_stamp_list_4[0:len(time_stamp_list_4) - 100]
                ether_account_balance_list = ether_account_balance_list[0:len(ether_account_balance_list) - 100]
                usd_eth_list = usd_eth_list[0:len(usd_eth_list) - 100]
                historic_usd_val_list = historic_usd_val_list[0:len(historic_usd_val_list) - 100]
                transaction_count_total_list = transaction_count_total_list[0:len(transaction_count_total_list) - 100]
                transaction_count_sent_list = transaction_count_sent_list[0:len(transaction_count_sent_list) - 100]
                transaction_count_recv_list = transaction_count_recv_list[0:len(transaction_count_recv_list) - 100]
                eth_fee_spent_list = eth_fee_spent_list[0:len(eth_fee_spent_list) - 100]
                eth_fee_used_list = eth_fee_used_list[0:len(eth_fee_used_list) - 100]
                token_transfers_list = token_transfers_list[0:len(token_transfers_list) - 100]
                outbound_transfers_list = outbound_transfers_list[0:len(outbound_transfers_list) - 100]
                inbound_transfers_list = inbound_transfers_list[0:len(inbound_transfers_list) - 100]
                unique_address_sent_list = unique_address_sent_list[0:len(unique_address_sent_list) - 100]
                unique_address_recv_list = unique_address_recv_list[0:len(unique_address_recv_list) - 100]
                token_contracts_count_list = token_contracts_count_list[0:len(token_contracts_count_list) - 100]
                ethereum_transactions_list = ethereum_transactions_list[0:len(ethereum_transactions_list) - 100]
                unique_outgoing_address_list = unique_outgoing_address_list[0:len(unique_outgoing_address_list) - 100]
                unique_incoming_address_list = unique_incoming_address_list[0:len(unique_incoming_address_list) - 100]
                ether_sent_out_list = ether_sent_out_list[0:len(ether_sent_out_list) - 100]
                ether_recv_in_list = ether_recv_in_list[0:len(ether_recv_in_list) - 100]


        temp = [a+b for a,b in zip(ether_sent_out_list, ether_recv_in_list)]
        transaction_amount_usd_origin = [a * b for a, b in zip(temp, usd_eth_list)]
        ether_account_balance_list_origin = ether_account_balance_list
        usd_eth_list_origin = usd_eth_list
        historic_usd_val_list_origin = historic_usd_val_list
        transaction_count_total_list_origin = transaction_count_total_list
        transaction_count_sent_list_origin = transaction_count_sent_list
        transaction_count_recv_list_origin = transaction_count_recv_list
        eth_fee_spent_list_origin = eth_fee_spent_list
        eth_fee_used_list_origin = eth_fee_used_list
        token_transfers_list_origin = token_transfers_list
        outbound_transfers_list_origin = outbound_transfers_list
        inbound_transfers_list_origin = inbound_transfers_list
        unique_address_sent_list_origin = unique_address_sent_list
        unique_address_recv_list_origin = unique_address_recv_list
        token_contracts_count_list_origin = token_contracts_count_list
        unique_outgoing_address_list_origin = unique_outgoing_address_list
        unique_incoming_address_list_origin = unique_incoming_address_list
        ether_sent_out_list_origin = ether_sent_out_list
        ether_recv_in_list_origin = ether_recv_in_list
        # 归一化
        if True:
            ether_account_balance_list = min_max_scaler(ether_account_balance_list)
            usd_eth_list = min_max_scaler(usd_eth_list)
            historic_usd_val_list = min_max_scaler(historic_usd_val_list)
            transaction_count_total_list = min_max_scaler(transaction_count_total_list)
            transaction_count_sent_list = min_max_scaler(transaction_count_sent_list)
            transaction_count_recv_list = min_max_scaler(transaction_count_recv_list)
            eth_fee_spent_list = min_max_scaler(eth_fee_spent_list)
            eth_fee_used_list = min_max_scaler(eth_fee_used_list)
            token_transfers_list = min_max_scaler(token_transfers_list)
            outbound_transfers_list = min_max_scaler(outbound_transfers_list)
            inbound_transfers_list = min_max_scaler(inbound_transfers_list)
            unique_address_sent_list = min_max_scaler(unique_address_sent_list)
            unique_address_recv_list = min_max_scaler(unique_address_recv_list)
            token_contracts_count_list = min_max_scaler(token_contracts_count_list)
            ethereum_transactions_list = min_max_scaler(ethereum_transactions_list)
            unique_outgoing_address_list = min_max_scaler(unique_outgoing_address_list)
            unique_incoming_address_list = min_max_scaler(unique_incoming_address_list)
            temp = [a+b for a,b in zip(ether_recv_in_list, ether_sent_out_list)]
            ether_sent_out_list = min_max_scaler(ether_sent_out_list)
            ether_recv_in_list = min_max_scaler(ether_recv_in_list)
            # transaction_amount_eth = min_max_scaler(temp)
            ether_usd_recv_in_list = min_max_scaler([a * b for a, b in zip(ether_recv_in_list, usd_eth_list)])
            ether_usd_sent_out_list = min_max_scaler([a * b for a, b in zip(ether_sent_out_list, usd_eth_list)])
            # ether_usd_total_list = min_max_scaler(temp)
            transaction_amount_usd = min_max_scaler([a * b for a, b in zip(temp, usd_eth_list)])
        # 画图
        if False:
            plt.figure()
            plt.suptitle(exchange)
            plt.subplot(2,2,1)
            plt.plot(time_stamp_list_1, ether_account_balance_list)
            plt.xlabel("timestamp")
            plt.ylabel("ETH")
            plt.title("ether_account_balance")
            plt.subplot(2,2,2)
            plt.plot(time_stamp_list_1, usd_eth_list)
            plt.xlabel("timestamp")
            plt.ylabel("USD/ETH")
            plt.title("usd_eth")
            plt.subplot(2,2,3)
            plt.plot(time_stamp_list_1, historic_usd_val_list)
            plt.xlabel("timestamp")
            plt.ylabel("USD")
            plt.title("historic_usd_val")
            plt.subplot(2,2,4)
            plt.plot(time_stamp_list_1, transaction_count_total_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("transaction_count_total")
            plt.show()
            plt.figure()
            plt.suptitle(exchange)
            plt.subplot(2,2,1)
            plt.plot(time_stamp_list_1, transaction_count_sent_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("transaction_count_sent")
            plt.subplot(2,2,2)
            plt.plot(time_stamp_list_1, transaction_count_recv_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("transaction_count_recv")
            plt.subplot(2,2,3)
            plt.plot(time_stamp_list_1, eth_fee_spent_list)
            plt.xlabel("timestamp")
            plt.ylabel("ETH")
            plt.title("eth_fee_spent")
            plt.subplot(2,2,4)
            plt.plot(time_stamp_list_1, eth_fee_used_list)
            plt.xlabel("timestamp")
            plt.ylabel("ETH")
            plt.title("eth_fee_used")
            plt.show()

            plt.figure()
            plt.suptitle(exchange)
            plt.subplot(3,2,1)
            plt.plot(time_stamp_list_2, token_transfers_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("token_transfers")
            plt.subplot(3, 2, 2)
            plt.plot(time_stamp_list_2, outbound_transfers_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("outbound_transfers")
            plt.subplot(3, 2, 3)
            plt.plot(time_stamp_list_2, inbound_transfers_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("inbound_transfers")
            plt.subplot(3, 2, 4)
            plt.plot(time_stamp_list_2, unique_address_sent_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("unique_address_sent")
            plt.subplot(3, 2, 5)
            plt.plot(time_stamp_list_2, unique_address_recv_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("unique_address_recv")
            plt.subplot(3, 2, 6)
            plt.plot(time_stamp_list_2, token_contracts_count_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("token_contracts_count")
            plt.show()

            plt.figure()
            plt.suptitle(exchange)
            plt.subplot(3,1,1)
            plt.plot(time_stamp_list_3, ethereum_transactions_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("ethereum_transactions")
            plt.subplot(3, 1, 2)
            plt.plot(time_stamp_list_3, unique_outgoing_address_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("unique_outgoing_address")
            plt.subplot(3, 1, 3)
            plt.plot(time_stamp_list_3, unique_incoming_address_list)
            plt.xlabel("timestamp")
            plt.ylabel("count")
            plt.title("unique_incoming_address")
            plt.show()

            plt.figure()
            plt.suptitle(exchange)
            plt.subplot(2, 1, 1)
            plt.plot(time_stamp_list_3, ether_sent_out_list)
            plt.xlabel("timestamp")
            plt.ylabel("ETH")
            plt.title("ether_sent_out")
            plt.subplot(2, 1, 2)
            plt.plot(time_stamp_list_3, ether_recv_in_list)
            plt.xlabel("timestamp")
            plt.ylabel("ETH")
            plt.title("ether_recv_in")
            plt.show()
        # transaction amount
        if True:
            # 宽一点的图
            length = 6
            width = 4.5
            # length = 10
            # width = 4.5
            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            plt.suptitle(exchange.title())
            x = range(len(time_stamp_list_1))
            plt.plot(x, transaction_amount_usd)
            plt.plot(x, ether_account_balance_list, '--')
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.legend(('transaction_amount_usd', 'account_balance_eth'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/'+exchange+'1.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            plt.suptitle(exchange.title())
            x = range(len(time_stamp_list_1))
            plt.plot(x, transaction_amount_usd)
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.plot(x, usd_eth_list, '--')
            plt.plot(x, historic_usd_val_list, '--')
            plt.legend(('transaction_amount_usd', 'ether_price_usd', 'account_balance_usd'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/'+exchange+'2.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            plt.suptitle(exchange.title())
            plt.plot(x, transaction_amount_usd)
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.plot(x, transaction_count_total_list, '--')
            plt.legend(('transaction_amount_usd', 'transaction_count_total'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/' + exchange + '3.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            plt.suptitle(exchange.title())
            plt.plot(x, transaction_amount_usd)
            plt.plot(x, transaction_count_sent_list, '--')
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.plot(x, transaction_count_recv_list, '--')
            plt.legend(('transaction_amount_usd', 'transaction_count_sent', 'transaction_count_recv'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/' + exchange + '4.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            plt.suptitle(exchange.title())
            plt.plot(x, transaction_amount_usd)
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.plot(x, eth_fee_spent_list, '--')
            plt.plot(x, eth_fee_used_list, '--')
            plt.legend(('transaction_amount_usd', 'fee_spent_eth', 'fee_used_eth'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/' + exchange + '5.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            x = range(len(time_stamp_list_2))
            plt.suptitle(exchange.title())
            plt.plot(x, transaction_amount_usd)
            plt.plot(x, token_transfers_list, '--')
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.legend(("transaction_amount_usd", 'token_transfers'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/' + exchange + '6.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            x = range(len(time_stamp_list_2))
            plt.suptitle(exchange.title())
            plt.plot(x, transaction_amount_usd)
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.plot(x, outbound_transfers_list, '--')
            plt.plot(x, inbound_transfers_list, '--')
            plt.legend(("transaction_amount_usd", 'outbound_transfers',
                        'inbound_transfers'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/' + exchange + '7.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            x = range(len(time_stamp_list_3))
            plt.suptitle(exchange.title())
            plt.plot(x, transaction_amount_usd)
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.plot(x, unique_address_sent_list, '--')
            plt.plot(x, unique_address_recv_list, '--')
            plt.legend(('transaction_amount_usd', 'unique_address_sent', 'unique_address_recv'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/' + exchange + '8.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            x = range(len(time_stamp_list_3))
            plt.suptitle(exchange.title())
            plt.plot(x, transaction_amount_usd)
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.plot(x, token_contracts_count_list, '--')
            plt.legend(('transaction_amount_usd', 'token_contracts_count'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/' + exchange + '9.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            x = range(len(time_stamp_list_3))
            plt.suptitle(exchange.title())
            plt.plot(x, transaction_amount_usd)
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.plot(x, unique_outgoing_address_list, '--')
            plt.plot(x, unique_incoming_address_list, '--')
            plt.legend(('transaction_amount_usd', 'unique_outgoing_address',
                        'unique_incoming_address'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/' + exchange + '10.png')
            plt.show()

            plt.figure(figsize=(length, width))
            plt.xlim(0, len(time_stamp_list_1))
            x = range(len(time_stamp_list_4))
            plt.suptitle(exchange.title())
            plt.plot(x, transaction_amount_usd)
            plt.plot(x, ether_sent_out_list, '--')
            plt.xlabel("时间", font1)
            plt.ylabel("归一化值", font1)
            plt.plot(x, ether_recv_in_list, '--')
            plt.legend(('transaction_amount_usd','ether_sent_out', 'ether_recv_in'), loc=2)
            plt.grid(linestyle="--")
            plt.savefig('./exchange/figure/' + exchange + '11.png')
            plt.show()
        # pearson相关系数
        if True:
            print("pearson相关系数")
            r, p = stats.pearsonr(ether_account_balance_list, transaction_amount_usd)
            print("ether_balance,           r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(usd_eth_list, transaction_amount_usd)
            print("usd_eth,                 r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(historic_usd_val_list, transaction_amount_usd)
            print("historic_usd_val,        r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(transaction_count_total_list, transaction_amount_usd)
            print("transaction_count,       r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(transaction_count_sent_list, transaction_amount_usd)
            print("transaction_count_sent,  r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(transaction_count_recv_list, transaction_amount_usd)
            print("transaction_count_recv,  r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(eth_fee_spent_list, transaction_amount_usd)
            print("eth_fee_spent,           r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(eth_fee_used_list, transaction_amount_usd)
            print("eth_fee_used,            r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(token_transfers_list, transaction_amount_usd)
            print("token_transfers,         r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(outbound_transfers_list, transaction_amount_usd)
            print("outbound_transfers,      r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(inbound_transfers_list, transaction_amount_usd)
            print("inbound_transfers,       r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(unique_address_recv_list, transaction_amount_usd)
            print("unique_address_recv,     r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(unique_address_sent_list, transaction_amount_usd)
            print("unique_address_sent,     r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(token_contracts_count_list, transaction_amount_usd)
            print("token_contracts_count,   r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(unique_outgoing_address_list, transaction_amount_usd)
            print("unique_outgoing_address, r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(unique_incoming_address_list, transaction_amount_usd)
            print("unique_incoming_address, r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(ether_sent_out_list, transaction_amount_usd)
            print("ether_sent_out,          r = {}, p ={}".format(r, p))
            r, p = stats.pearsonr(ether_recv_in_list, transaction_amount_usd)
            print("ether_recv_in,           r = {}, p ={}".format(r, p))

        # spearman相关系数
        if True:
            print("spearman相关系数")
            r, p = stats.spearmanr(ether_account_balance_list, transaction_amount_usd)
            print("ether_balance,           r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(usd_eth_list, transaction_amount_usd)
            print("usd_eth,                 r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(historic_usd_val_list, transaction_amount_usd)
            print("historic_usd_val,        r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(transaction_count_total_list, transaction_amount_usd)
            print("transaction_count,       r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(transaction_count_sent_list, transaction_amount_usd)
            print("transaction_count_sent,  r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(transaction_count_recv_list, transaction_amount_usd)
            print("transaction_count_recv,  r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(eth_fee_spent_list, transaction_amount_usd)
            print("eth_fee_spent,           r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(eth_fee_used_list, transaction_amount_usd)
            print("eth_fee_used,            r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(token_transfers_list, transaction_amount_usd)
            print("token_transfers,         r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(outbound_transfers_list, transaction_amount_usd)
            print("outbound_transfers,      r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(inbound_transfers_list, transaction_amount_usd)
            print("inbound_transfers,       r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(unique_address_recv_list, transaction_amount_usd)
            print("unique_address_recv,     r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(unique_address_sent_list, transaction_amount_usd)
            print("unique_address_sent,     r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(token_contracts_count_list, transaction_amount_usd)
            print("token_contracts_count,   r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(unique_outgoing_address_list, transaction_amount_usd)
            print("unique_outgoing_address, r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(unique_incoming_address_list, transaction_amount_usd)
            print("unique_incoming_address, r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(ether_sent_out_list, transaction_amount_usd)
            print("ether_sent_out,          r = {}, p ={}".format(r, p))
            r, p = stats.spearmanr(ether_recv_in_list, transaction_amount_usd)
            print("ether_recv_in,           r = {}, p ={}".format(r, p))

            print("exchange = {}".format(exchange))
            minTime = min(time_stamp_list_1)
            maxTime = max(time_stamp_list_1)
            timeArray = time.localtime(minTime)
            otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            print("最小时间戳 = {}，最小时间 = {}".format(minTime, otherStyleTime))
            timeArray = time.localtime(maxTime)
            otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            print("最大时间戳 = {}，最大时间 = {}".format(maxTime, otherStyleTime))
            print("相隔{}天".format((maxTime-minTime)/86400+1))

        # 储存选取的指标
        if True:
            # 创建feature文件
            if exchange=="binance":
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'usd_eth': usd_eth_list_origin,
                                   'historic_usd_val': historic_usd_val_list_origin, 'transaction_count_total': transaction_count_total_list_origin,
                                    'transaction_count_sent': transaction_count_sent_list_origin, 'eth_fee_spent': eth_fee_spent_list_origin,
                                    'token_transfers': token_transfers_list_origin, 'outbound_transfers': outbound_transfers_list_origin,
                                    'inbound_transfers': inbound_transfers_list_origin, 'unique_address_recv': unique_address_recv_list_origin,
                                    'unique_address_sent': unique_address_sent_list_origin,
                                    'token_contracts_count': token_contracts_count_list_origin, 'unique_outgoing_address': unique_outgoing_address_list_origin,
                                    'ether_sent_out': ether_sent_out_list_origin, 'ether_recv_in': ether_recv_in_list_origin})
                df.to_csv('./exchange/feature/binance_ft.csv', index=False)
            if exchange=="coinbase":
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'usd_eth': usd_eth_list_origin,
                                   'transaction_count': transaction_count_total_list_origin, 'transaction_count_sent': transaction_count_sent_list_origin,
                                   'transaction_count_recv': transaction_count_recv_list_origin, 'eth_fee_spent':eth_fee_spent_list_origin,
                                   'eth_fee_used': eth_fee_used_list_origin, 'token_transfers': token_transfers_list_origin,
                                   'outbound_transfers': outbound_transfers_list_origin, 'unique_address_sent': unique_address_sent_list_origin,
                                   'unique_outgoing_address': unique_outgoing_address_list_origin, 'ether_sent_out': ether_sent_out_list_origin,
                                   'ether_recv_in': ether_recv_in_list_origin})
                df.to_csv('./exchange/feature/coinbase_ft.csv', index=False)
            if exchange=="huobi":
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'usd_eth': usd_eth_list_origin,
                                   'historic_usd_val': historic_usd_val_list_origin, 'eth_fee_used': eth_fee_used_list_origin,
                                   'ether_sent_out': ether_sent_out_list_origin})
                df.to_csv('./exchange/feature/huobi_ft.csv', index=False)
            if exchange=='kraken':
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'usd_eth':usd_eth_list_origin,
                                   'transaction_count': transaction_count_total_list_origin, 'transaction_count_sent': transaction_count_sent_list_origin,
                                   'transaction_count_recv': transaction_count_recv_list_origin, 'eth_fee_spent': eth_fee_spent_list_origin,
                                   'eth_fee_used': eth_fee_used_list_origin, 'token_tranfers': token_transfers_list_origin,
                                   'inbound_transfers': inbound_transfers_list_origin, 'unique_address_recv': unique_address_recv_list_origin,
                                   'token_contracts_count': token_contracts_count_list_origin, 'unique_outgoing_address': unique_outgoing_address_list_origin,
                                   'unique_incoming_address': unique_incoming_address_list_origin})
                df.to_csv('./exchange/feature/kraken_ft.csv', index=False)
            if exchange=='kucoin':
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'usd_eth': usd_eth_list_origin,
                                   'transaction_count': transaction_count_total_list_origin, 'transaction_count_sent': transaction_count_sent_list_origin,
                                   'eth_fee_spent': eth_fee_spent_list_origin, 'eth_fee_used': eth_fee_used_list_origin,
                                   'token_transfers': token_transfers_list_origin, 'outbound_transfers': outbound_transfers_list_origin,
                                   'inbound_transfers': inbound_transfers_list_origin, 'unique_address_recv': unique_address_recv_list_origin,
                                   'unique_address_sent': unique_address_sent_list_origin, 'unique_outgoing_address': unique_outgoing_address_list_origin})
                df.to_csv('./exchange/feature/kucoin_ft.csv', index=False)
        if True:
            # 创建不重要feature文件
            if exchange=="binance":
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'account_balance': ether_account_balance_list_origin,
                                   'transaction_count_recv': transaction_count_recv_list_origin, 'unique_incoming_address': unique_incoming_address_list_origin})
                df.to_csv('./exchange/feature/binance_ft_not_important.csv', index=False)
            if exchange=="coinbase":
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'account_balace': ether_account_balance_list_origin,
                                   'historic_usd_val': historic_usd_val_list_origin, 'inbound_transfers': inbound_transfers_list_origin,
                                   'unique_address_recv': unique_address_recv_list_origin, 'token_contracts_count': token_contracts_count_list_origin,
                                   'unique_incoming_address': unique_incoming_address_list_origin})
                df.to_csv('./exchange/feature/coinbase_ft_not_important.csv', index=False)
            if exchange=="huobi":
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'account_balance': ether_account_balance_list_origin,
                                   'transaction_count': transaction_count_total_list_origin, 'transaction_count_sent': transaction_count_sent_list_origin,
                                   'transaction_count_recv': transaction_count_recv_list_origin, 'eth_fee_spent': eth_fee_spent_list_origin,
                                   'token_transfers': token_transfers_list_origin, 'outbound_transfers': outbound_transfers_list_origin,
                                   'inbound_transfers': inbound_transfers_list_origin, 'unique_address_recv': unique_address_recv_list_origin,
                                   'unique_address_sent': unique_address_sent_list_origin, 'token_contracts_count': token_contracts_count_list_origin,
                                   'unique_outgoing_address': unique_outgoing_address_list_origin, 'unique_incoming_address': unique_incoming_address_list_origin,
                                   'ether_recv_in': ether_recv_in_list_origin})
                df.to_csv('./exchange/feature/huobi_ft_not_important.csv', index=False)
            if exchange=='kraken':
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'account_balance': ether_account_balance_list_origin,
                                   'outbound_transfers': outbound_transfers_list_origin, 'unique_address_sent': unique_address_sent_list_origin,
                                   'ether_sent_out': ether_sent_out_list_origin, 'ether_recv_in': ether_recv_in_list_origin})
                df.to_csv('./exchange/feature/kraken_ft_not_important.csv', index=False)
            if exchange=='kucoin':
                df = pd.DataFrame({'transaction_amount_usd': transaction_amount_usd_origin, 'account_balance': ether_account_balance_list_origin,
                                   'transaction_count_recv': transaction_count_recv_list_origin, 'token_contracts_count': token_contracts_count_list_origin,
                                   'unique_incoming_address':unique_incoming_address_list_origin, 'ether_sent_out': ether_sent_out_list_origin,
                                   'ether_recv_in': ether_recv_in_list_origin})
                df.to_csv('./exchange/feature/kucoin_ft_not_important.csv', index=False)
if __name__=='__main__':
    # save_features()
    load_features()

