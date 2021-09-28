#
#
# import json
# with open("/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/save/r50.txt", "r") as fp:
#    b = json.load(fp)

# import xlsxwriter
#
# workbook = xlsxwriter.Workbook('/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/save/r50.xlsx')
# worksheet = workbook.add_worksheet()
#
# my_list = b
#
# for row_num, row_data in enumerate(my_list):
#     for col_num, col_data in enumerate(row_data):
#         worksheet.write(row_num, col_num, col_data)
#
# workbook.close()

#


import xlsxwriter
import os

# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/all_images.xlsx')
worksheet = workbook.add_worksheet()

test_image = '/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/test'
image_hr = '/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/save/hrnet'
image_r18 = '/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/save/r18'
image_r50 = '/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/save/r50'
image_dense = '/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/save/densenet'


i=0

for folder in os.listdir(test_image):

    print(folder)

    imglist = os.listdir(os.path.join(test_image,folder))

    print(imglist)

    for img in imglist:

        i += 1

        src_hr = os.path.join(image_hr, folder, img)
        src_r18 = os.path.join(image_r18, folder, img)
        src_r50 = os.path.join(image_r50, folder, img)
        src_dense = os.path.join(image_dense, folder, img)

        worksheet.write('A' + str(i), img)
        worksheet.write('B' + str(i), folder)
        worksheet.insert_image('C' + str(i), src_hr, {'x_scale': 0.25, 'y_scale': 0.25})
        worksheet.insert_image('D' + str(i), src_r18, {'x_scale': 0.25, 'y_scale': 0.25})
        worksheet.insert_image('E' + str(i), src_r50, {'x_scale': 0.25, 'y_scale': 0.25})
        worksheet.insert_image('F' + str(i), src_dense, {'x_scale': 0.25, 'y_scale': 0.25})

workbook.close()


