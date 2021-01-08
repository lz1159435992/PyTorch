import xlwt
import xlrd
import numpy as np
def ge_excel(filepath):
    wb = xlwt.Workbook()
    # 添加一个表
    ws = wb.add_sheet('test')
    style = xlwt.XFStyle()
    style.num_format_str = '0.000%'
    report = np.load(filepath+'\\report\\report.npy').item()
    failed_num = np.load(filepath+'\\report\\failed_num.npy').item()
    print(failed_num)
    i = 0
    for k,v in report.items():
        # 3个参数分别为行号，列号，和内容
        # 需要注意的是行号和列号都是从0开始的
        if i == 0 :
            ws.write(i, 0, '变异体名字')
            ws.write(i, 1, '排名')
            ws.write(i, 2, 'EXAM')
            ws.write(i, 3, '失败用例个数')
            i = i + 1
        ws.write(i, 0, k)
        ws.write(i, 1, v[0])
        ws.write(i, 2, v[1], style)
        ws.write(i, 3, failed_num[k])
        i = i + 1
    wb.save(filepath+'\\report\\test.xls')
def ge_excel_forward(filepath):
    wb = xlwt.Workbook()
    # 添加一个表
    ws = wb.add_sheet('test')
    style = xlwt.XFStyle()
    style.num_format_str = '0.000%'
    report = np.load(filepath+'\\report_forward\\report.npy').item()
    failed_num = np.load(filepath+'\\report\\failed_num.npy').item()
    print(failed_num)
    i = 0
    for k,v in report.items():
        # 3个参数分别为行号，列号，和内容
        # 需要注意的是行号和列号都是从0开始的
        if i == 0 :
            ws.write(i, 0, '变异体名字')
            ws.write(i, 1, '排名')
            ws.write(i, 2, 'EXAM')
            ws.write(i, 3, '失败用例个数')
            i = i + 1
        ws.write(i, 0, k)
        ws.write(i, 1, v[0])
        ws.write(i, 2, v[1], style)
        ws.write(i, 3, failed_num[k])
        i = i + 1
    wb.save(filepath+'\\report_forward\\test.xls')
def ge_excel_final(filepath):
    wb = xlwt.Workbook()
    # 添加一个表
    ws = wb.add_sheet('test')
    style = xlwt.XFStyle()
    style.num_format_str = '0.000%'
    report = np.load(filepath+'\\report\\report.npy').item()
    report_forward = np.load(filepath + '\\report_forward\\report.npy').item()
    report_final = np.load(filepath + '\\report_final\\report.npy').item()
    failed_num = np.load(filepath+'\\report\\failed_num.npy').item()
    print(failed_num)
    i = 0
    for k,v in report.items():
        # 3个参数分别为行号，列号，和内容
        # 需要注意的是行号和列号都是从0开始的
        if i == 0 :
            ws.write(i, 0, '变异体名字')
            ws.write(i, 1, '反向排名')
            ws.write(i, 2, '反向EXAM')
            ws.write(i, 3, '正向排名')
            ws.write(i, 4, '正向EXAM')
            ws.write(i, 5, '最终排名')
            ws.write(i, 6, '最终EXAM')
            ws.write(i, 7, '失败用例个数')
            i = i + 1
        ws.write(i, 0, k)
        ws.write(i, 1, v[0])
        ws.write(i, 2, v[1], style)
        ws.write(i, 3, report_forward[k][0])
        ws.write(i, 4, report_forward[k][1], style)
        ws.write(i, 5, report_final[k][0])
        ws.write(i, 6, report_final[k][1], style)
        ws.write(i, 7, failed_num[k])
        i = i + 1
    wb.save(filepath+'\\report_final\\test.xls')
if __name__ == '__main__':
    ge_excel()
