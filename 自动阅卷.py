#coding=gbk
import os
import xlwt
import codecs
import re

file_list = []

def set_style(name,height,color,bold = False, italic = False):
    style = xlwt.XFStyle() # ��ʼ����ʽ

    font = xlwt.Font() # Ϊ��ʽ��������
    font.name = name # 'Times New Roman'
    font.bold = bold #����
    font.colour_index = color #������ɫ
    font.height = height #�����С, 240��Ӧ��12����200��Ӧ10������12����ϵ
    font.italic = italic #б��

    style.font = font
    
    return style

def vstdir(path,fltp = '.txt'):
    for x in os.listdir(path):
        sub_path = os.path.join(path,x) 
        if os.path.isdir(sub_path):
            vstdir(sub_path)
        else:
            if fltp in sub_path.lower():
                file_list.append(sub_path)

def read_ans(path):
    f = open(x)
    ans_sheet = f.read()
    sheet_list = ans_sheet.split('---------------------------------------------------')

    #��ȡ������ѧ������
    title = sheet_list[0]

    d_id = p_did.findall(title)[0]
    s_id = p_sid.findall(title)[0]
    s_name = p_name.findall(title)[0]
    
    t_dict = {u'ѧ��':s_id[1].strip(),u'����':d_id[1].strip(),u'����':s_name[1].strip()} #ѧ�ţ����ź��������л��з�

    #��ȡ��
    ans_1 = sheet_list[2].strip()
    ans_2 = sheet_list[4].strip()
    ans_3 = sheet_list[6].strip()
    ans_4 = sheet_list[8].strip()
    ans_5 = sheet_list[10].strip()
    ans_6 = sheet_list[12].strip()
    ans_7 = sheet_list[14].split('-------------------------------')
    ans_7_1 = ans_7[2].strip()
    ans_7_2 = ans_7[4].strip()
    
    t_dict['1'] = p_ans.findall(ans_1)
    t_dict['2'] = p_ans.findall(ans_2)
    t_dict['3'] = p_ans.findall(ans_3)
    t_dict['4'] = p_ans.findall(ans_4)
    t_dict['5'] = p_ans.findall(ans_5)
    t_dict['6'] = p_ans.findall(ans_6)
    t_dict['7_1'] = p_ans.findall(ans_7_1)
    t_dict['7_2'] = p_ans.findall(ans_7_2)

    return t_dict


path = '/Users/xiaoyu/Desktop/3-56'
vstdir(path)

p_name = re.compile('\xe5\xa7\x93\xe5\x90\x8d:(\s*\S+)')
p_sid = re.compile('\xe5\xad\xa6\xe5\x8f\xb7:(\s*\d+)')
p_did = re.compile('\xe6\x9c\xba\xe5\x8f\xb7:(\s*\d+)')
p_ans = re.compile('^\d.(\S+)',re.M)

m3_56 = {}
    
for x in file_list:
    t_dict = read_ans(x)
    m3_56[t_dict[u'ѧ��']] = t_dict
   
 



style1 = set_style('Times New Roman',200,0)
style2 = set_style('Times New Roman',200,2,bold = True)

wb = xlwt.Workbook(encoding='utf-8')
tb1 = wb.add_sheet('Choice',cell_overwrite_ok = True)
tb2 = wb.add_sheet('Blanket',cell_overwrite_ok = True)
tb3 = wb.add_sheet('Procedure',cell_overwrite_ok = True)
tb4 = wb.add_sheet('Summary',cell_overwrite_ok = True)

title_list = [u'ѧ��',u'����',u'����',u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9',u'10',u'��ȷ����']

for j in range(14):
    tb1.write(0,j,title_list[j])
    tb2.write(0,j,title_list[j])

for j in range(9):
    tb3.write(0,j,title_list[j])

tb3.write(0,9,u'��ȷ����')

for j in range(3):
    tb4.write(0,j,title_list[j])
    
tb4.write(0,3,u'ѡ����')
tb4.write(0,4,u'�����')
tb4.write(0,5,u'������')
tb4.write(0,6,u'�ܼ�')

i = 1
sum_list1 = [0]*10 #�ж���λѧ������ѡ����
sum_list2 = [0]*10 #�ж���λѧ�����������

w = codecs.open('/Users/xiaoyu/Desktop/3_56_procedures.txt','w','utf-8')

for k in sorted(m3_56.keys()):
    s = m3_56[k]
    print int(s[u'����'])
    w_list1 = [s[u'ѧ��'],s[u'����'],s[u'����']]+s[u'ѡ����']
    w_list2 = [s[u'ѧ��'],s[u'����'],s[u'����']]+s[u'�����']
    
    #�ѳ���д��txt�ĵ�
    t1 = s[u'ѧ��']+s[u'����']+s[u'����']
    w.write(u'%s--%d--%s\n\n'%(s[u'ѧ��'],int(s[u'����']),s[u'����']))
    for t in range(5):
        w.write(u'%s--��%d��\n'%(s[u'ѧ��'],t+1))
        w.write(s[u'������'][t])
        w.write('\n-----------------------------------------------------------\n')
    
    w.write('\n*********************************************************\n')
    
    #��ѡ����������д��excel�ļ�����ʵ����ȷ����ͳ��
    for j in range(3):
        tb1.write(i,j,w_list1[j],style1)
        tb2.write(i,j,w_list2[j],style1)
        tb3.write(i,j,w_list2[j],style1)
        tb4.write(i,j,w_list2[j],style1)

    count1 = 0
    count2 = 0
    for j in range(10):
        if daan_1[j] == w_list1[j+3]:
            tb1.write(i,j+3,w_list1[j+3],style1)
            count1 += 1
            sum_list1[j] += 1
            
        else:
            tb1.write(i,j+3,w_list1[j+3],style2)

        if w_list2[j+3] in daan_2[j]:
            tb2.write(i,j+3,w_list2[j+3],style1)
            count2 += 1
            sum_list2[j] += 1
            
        else:
            tb2.write(i,j+3,w_list2[j+3],style2)
            
    
    tb1.write(i,13,count1)
    tb2.write(i,13,count2)
    
    tb4.write(i,3,count1)
    tb4.write(i,4,count2)

    i += 1

for j in range(10):
    tb1.write(i,j+3,sum_list1[j],style1)
    tb2.write(i,j+3,sum_list2[j],style1)

wb.save('/Users/xiaoyu/Desktop/3-56/3_56.xls')
w.close()