#coding=utf-8
score = [5, 8, 7, 4, -1]
len_s = len(score)

for i in range(len_s - 1):
    min_idx = i
    print '第' + str(i + 1) + '轮:' + str(score[i:])
    for j in range(i + 1, len_s):
        if score[min_idx] > score[j]:
            min_idx = j

    score[min_idx], score[i] = score[i], score[min_idx]
    print