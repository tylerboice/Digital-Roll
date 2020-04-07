import base64
from PIL import Image

def importer(picture, xAccel, yAccel, zAccel, dshape, dvalue, bboxminx, bboxminy, bboxmaxx, bboxmaxy):
    #file = open("test.xml", "w")
    try:
        shapestring = dshape[1:]
    except:
        print("dice shape is funky")
    if dvalue <= int(shapestring):
        pass
    else:
        raise Exception("dice value is invalid for dice shape")
    if isinstance(dvalue, int):
        dvalue = str(dvalue)
    if picture.endswith('.jpg'):
        pass
    else:
        raise Exception("picture is in invalid format")
    if -1 <= xAccel <= 1:
        pass
    else:
        raise Exception("accelerometer X is invalid")
    if -1 <= yAccel <= 1:
        pass
    else:
        raise Exception("accelerometer X is invalid")
    if -1 <= zAccel <= 1:
        pass
    else:
        raise Exception("accelerometer X is invalid")
    try:
        xvec = str(xAccel)
    except:
        print("could not convert xAccel to string")
    try:
        yvec = str(yAccel)
    except:
        print("could not convert yAccel to string")
    try:
        zvec = str(zAccel)
    except:
        print("could not convert zAccel to string")
    if bboxminx > bboxmaxx or bboxminy > bboxmaxy :
        raise Exception("invalid bounding box parameters")
    convert(picture, xvec, yvec, zvec, dshape, dvalue, bboxminx, bboxminy, bboxmaxx, bboxmaxy)

def convert(image, xvec, yvec, zvec, dshape, dvalue, bboxminx, bboxminy, bboxmaxx, bboxmaxy):
    try:
        converted_file = open( dshape + "-" + dvalue + ".xml", "w")
    except:
        print("failed to open requested file")

    try:
        pilImage = Image.open(image)
    except:
        print("image could not be opened properly")
    try:
        width, height = pilImage.size
    except:
        print("size could not be assigned properly")

    try:
        widthstr = str(width)
    except:
        print("could not convert width to string")
    try:
        heightstr = str(height)
    except:
        print("could not convert height to string")
    try:
        with open(image, "rb") as img_file:
            imgHandler = img_file.read()
    except:
        print("image could not be read as bytes")
    try:
        converted_string = base64.b64encode(imgHandler)
    except:
        print("the image could not be converted to base 64")
    #file_text = "<annotation>\n <filename>\ " converted_file.filename \ "</filename>\n <img>\ " converted_string \ "</img>\n <xAccel>\ " xvec \ "</xAccel>\n <yAccel>\ " yvec \ "</yAccel>\n <zAccel>\ " zvec \ "</zAccel>\n</annotation>"
    try:
        converted_file.write("<annotation>\n <filename>" + converted_file.name + "</filename>\n <img>" + str(converted_string) + "</img>\n " +
        " <xAccel>" + xvec + "</xAccel>\n <yAccel>" + yvec + "</yAccel>\n <zAccel>" + zvec + "</zAccel>\n" +
        " <size>\n \t<width>" + widthstr + "</width> \n <height>" + heightstr + "</height> \n<depth>3</depth> \n</size>"
        + "<object> \n \t <name>" + dshape + "-" + dvalue + "</name>\n <pose>Unspecified</pose> \n <truncated>0</truncated>\n <difficult>0</difficult>\n "
        + "<bndbox> \n \t <xmin>" + str(bboxminx) + "</xmin>\n <ymin>" + str(bboxminy) + "</ymin> \n <xmax>" + str(bboxmaxx) + "</xmax> \n <ymax>" + str(bboxmaxy) + " </ymax> \n </bndbox> \n </object>"
        + "</annotation>")
    except:
        print("could not write requested string to file")
    finally:
        converted_file.close()

    output(converted_file)

def output(output_file):
    return output_file

#testpic = open("snowwhite.jpg", "rb") as img_file
#file = open("test.xml", "w")
#file.close
#im = Image.open(r"/snowwhite")
importer("snowwhite.jpg", 0, 0, 0, 'd10', 10, 180, 315, 396, 508)
