import numpy as np
import os
import subprocess
import sys
import select
from time import sleep

#ファイルパス
KAGO_FILE_PATH = "./sys/kago.csv"
#入力タイムアウト
DEFAULT_TIMEOUT = 1
#MAIN_LOOP
MAIN_LOOP_SLEEP = 1

#Display
COL1_LENGTH = 7
COL2_LENGTH = 20
COL3_LENGTH = 7

#s = str
"""
bar1 = [s for _ in range(COL1_LENGTH)]
bar2 = [s for _ in range(COL2_LENGTH)]
bar3 = [s for _ in range(COL3_LENGTH)]
"""

def unix_input_with_timeout(prompt='', timeout=DEFAULT_TIMEOUT):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    (ready, _, _) = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().rstrip('\n')
    else:
        pass


def clear_display():
    try:
        subprocess.check_call('clear')
    except:
        print("Command execution error. [clear]")


def display_item_w_less():
    cmd = "less {}".format(KAGO_FILE_PATH)
    try:
        print(os.popen(cmd).read())
    except:
        print("Command execution error. [{}]".format(cmd))


def display_kago():
    #kagoファイルをopen
    with open(KAGO_FILE_PATH, mode='r+') as f:
        #選択したitemをファイルから削除
        read_line = f.readlines()
        f.seek(0)
        print("{}|{}|{}".format("No.".center(COL1_LENGTH), "Item".center(COL2_LENGTH), "Price".center(COL3_LENGTH)))
        print("-------|--------------------|-------")
        #print("{}|{}|{}".format(bar1, bar2, bar3))
        sum = 0
        for i, line in enumerate(read_line):
            #lineを","で分解
            line = line.rstrip('\n')
            tmp = line.split(",")
            sum = sum + (int(tmp[2]))
            print("{}|{}|{}".format(tmp[0].center(COL1_LENGTH), tmp[1].center(COL2_LENGTH), tmp[2].center(COL3_LENGTH)))

        #print("{}|{}|{}".format(bar1, bar2, bar3))
        print("------------------------------------")
        str_sum = str(sum)
        str_sum = "Total : {}".format(str_sum).rjust(COL1_LENGTH + COL2_LENGTH + COL3_LENGTH)
        print("\n{}\n".format(str_sum))

    return True


def proceed_payment():
    #カゴの中身を表示 & 確認
    clear_display()
    print("Please confirm scanned item list below")
    print("\n")
    display_kago()
    id = input("0:OK 1:Back to scan\n")
    if id is str("1"):
        return False

    #データを送信（試作ではキーボードの入力待ち）
    input("Please press any key\n")

    #kagoファイルをopen & kagoファイルの中身をクリア
    with open(KAGO_FILE_PATH, mode='w') as f:
        f.write("")

    return True


def clear_item():
    #itemを選択
    #カゴの中身を表示 & 確認
    clear_display()
    display_kago()
    print("\n")
    item_id = input("Please select the Item No. you want to cancel\n or press '0' and go back to previous\n")

    str_item_id = ""
    for i in range(len(item_id)):
        str_item_id += "{}".format(item_id[i])
        #print("i:{} id:{}".format(i, str_item_id))

    #kagoファイルをopen
    with open(KAGO_FILE_PATH, mode='r+') as f:
        #選択したitemをファイルから削除
        read_line = f.readlines()
        f.seek(0)
        new_index = 1
        for i, line in enumerate(read_line):
            #lineを","で分解
            tmp = line.split(",")

            #削除したい行以外を書き戻す。（indexは上から振り直す）
            id_str = "{}".format(i+1)
            if id_str == str_item_id:
                continue

            new_line = "{},{},{}".format(new_index, tmp[1], tmp[2])
            f.write(new_line)
            new_index = new_index + 1

        f.truncate()

    return True


# Main Loop
if __name__ == '__main__':
    #画面をクリア
    clear_display()

    prev_file_time = 0
    while True:
        # キーボード入力を読み込み(タイムアウト付き)
        process_id = None
        process_id = unix_input_with_timeout()

        # 処理0
        if process_id is str("0"):
            ret = proceed_payment()
            if ret == False:
                prev_file_time = 0
                #clear_display()
                #display_kago()
                # print("Error!! Can't proceed payment")

        # 処理1
        elif process_id is str("1"):
            ret = clear_item()
            if ret == False:
                prev_file_time = 0
                # print("Error!! Can't cancel")
                #clear_display()
                #display_kago()
        else:
            # ilegal input, refresh display
            prev_file_time = 0

        file_time = os.stat(KAGO_FILE_PATH).st_mtime
        if prev_file_time != file_time:
            prev_file_time = file_time

            clear_display()
            display_kago()

            print("Please continue item scan or select a process you want")
            print("0:Proceed payment  1:cancel item from scanned list")

        sleep(MAIN_LOOP_SLEEP)
