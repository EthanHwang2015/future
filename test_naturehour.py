#!/usr/bin/python3
 
import pymongo
from datetime import datetime, time
import pdb
import sys
import argparse


def datetime2str(dt):
    return dt.strftime("%Y/%m/%d %H:%M")

def str2datetime(str):
    return datetime.strptime(str, "%Y/%m/%d %H:%M")


 
def gen_start_end_time(date, now_time):
    start = end = None
    day_hour_start = [time(9,00), time(10,00), time(11,00), time(13,00), time(14,00), time(21,00), time(22,00)]
    day_hour_end = [time(9,59), time(10,59), time(11,29), time(13,59), time(14,59), time(21,59), time(22,59)]
    if now_time < day_hour_start[0] or now_time > day_hour_start[-1]:
        start, end = (day_hour_start[-1], day_hour_end[-1])
    else:
        for i,h in enumerate(day_hour_start):
            if now_time <= h:
                print i,h
                start, end = (day_hour_start[i-1], day_hour_end[i-1])
                break

    str_start = date.strftime("%Y/%m/%d")+ " " +start.strftime("%H:%M")
    str_end = date.strftime("%Y/%m/%d")+ " " +end.strftime("%H:%M")
    return str2datetime(str_start), str2datetime(str_end)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='load data fron mongodb')
    parser.add_argument('--data', required=True, default="i1909",help='i1909')
    parser.add_argument('--appendFile', required=True, default="./seq_i9888_60min.csv",help='seq_i9888_60min')

    params, ukown = parser.parse_known_args(sys.argv[1:])
    dataset = params.data
    append_file = params.appendFile


    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["VnTrader_1Min_Db"]
    mycol = mydb[dataset]

    now = datetime.now()
        
     
    #datetime.timedelta(days=1)
    nowday = now.date()
    now_time = now.time()
#    now_time=time(12,30)
#    nowday = datetime.strptime("2019/06/21", "%Y/%m/%d")
    start , end = gen_start_end_time(nowday, now_time)


    datas = mycol.find({"datetime":{"$gte":start, "$lte":end}})
    open_price = 0
    high_price = 0
    low_price = 99999
    close_price = 0
    volume = 0
    open_interest = 0

#    print datas.count()
    print "now:{}, start:{}, end:{}, count:{}".format(now,start,end, datas.count())
    for x in datas:
        dt = x['datetime']
        if dt < start or dt > end:
            continue
        if open_price < 1:
            open_price = float(x['open'])
        high_price = max(float(x['high']), high_price)
        low_price = min(float(x['low']), low_price)
        close_price = float(x['close'])
        volume += int(x['volume'])
        open_interest = int(x['openInterest'])
    with open(append_file, 'a') as f:
        line = "{},{},{},{},{},{},{}\n".format(datetime2str(start), open_price, high_price, low_price, close_price, volume, open_interest)
        print line
        f.write(line)
     
#print date
#break
