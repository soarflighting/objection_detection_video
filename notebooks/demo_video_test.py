from notebooks.demo_video import *

if __name__ == '__main__':
    path = 'd:/input_data/threepeople.jpg'
    # 读取图片
    img = mpimg.imread(path)
    # 执行主流程函数
    image = process_image(img)
    plt.imshow(image)
    plt.show()