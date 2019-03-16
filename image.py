

from PIL import Image
import numpy as np
import os
def resizeImg(path):
    try:
        im = Image.open(path)
        # reshape image
        x_s = 50
        y_s = 50
        out = im.resize((x_s, y_s), Image.ANTIALIAS)
        im2 = np.array(out)
        imx = np.ndarray([im2.shape[0], im2.shape[1], 3])
        # fix the channels of black and white images
        if (len(im2.shape) < 3):

            for x in range(im2.shape[0]):
                for y in range(im2.shape[1]):
                    imx[x][y][0] = im2[x][y]
                    imx[x][y][1] = im2[x][y]
                    imx[x][y][2] = im2[x][y]
        elif(im2.shape[2] !=3):
            for x in range(im2.shape[0]):
                for y in range(im2.shape[1]):
                    imx[x][y][0] = im2[x][y][0]
                    imx[x][y][1] = im2[x][y][1]
                    imx[x][y][2] = im2[x][y][2]

        else:
            imx = im2
        return imx

    except Exception as e:
        print(e)

        return None

    return None
def resizepurdue():
    purdueset=[]
    pathlist=os.listdir('image/purdue logo')
    for path in pathlist:
        img = resizeImg('image/purdue logo/'+path)

        if isinstance(img,np.ndarray):
            try:
                # im=Image.fromarray(np.asarray(img))
                # im.save('resized_img/'+path)
                purdueset.append(img)
                # print(img.shape)
            except Exception as e:
                print(e.with_traceback())
    np.save(file='pu.npy',arr=purdueset)
    print(len(purdueset))
    return purdueset
def resizeIU():
    iuset=[]
    pathlist=os.listdir('image/indiana university logo')
    for path in pathlist:
        img = resizeImg('image/indiana university logo/'+path)

        if isinstance(img,np.ndarray):
            try:
                # im=Image.fromarray(np.asarray(img))
                # im.save('resized_img/'+path)
                iuset.append(img)
                # print(img.shape)
            except Exception as e:
                print(e.with_traceback())
    np.save(file='iu.npy',arr=iuset)
    print(len(iuset))
    return purdueset


def trainsetGen():
    puarr=np.load('pu.npy')
    iuarr=np.load('iu.npy')
    dataset=[]
    onehot_lable=[]
    for pu in puarr:
        dataset.append(pu)
        onehot_lable.append([1,0])
    for iu in iuarr:
        dataset.append(iu)
        onehot_lable.append([0,1])
    dataset=np.array(dataset)
    onehot_lable=np.array(onehot_lable)
    print(dataset.shape)
    print(onehot_lable.shape)
    np.save(file='x.npy',arr=dataset)
    np.save(file='y.npy',arr=onehot_lable)
if __name__=='__main__':
    purdueset=resizepurdue()
    inset=resizeIU()
    trainsetGen()