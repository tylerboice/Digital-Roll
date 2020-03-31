import base64

def importer(picture, xAccel, yAccel, zAccel):
    #file = open("test.xml", "w")
    image = picture
    try:
        xvec = str(xAccel)
    except:
        print("could not convert xAccel to string")
    try:
        yvec = str(yAccel)
    except:
        print("could not convert yAccel to string")
    try:
        zvec = (zAccel)
    except:
        print("could not convert zAccel to string")
    convert(image, xvec, yvec, zvec)

def convert(image, xvec, yvec, zvec):
    try:
        converted_file = open("test.xml", "w")
    except:
        print("failed to open requested file")
    with open(image, "rb") as img_file:
        imgHandler = img_file.read()
    try:
        converted_string = base64.b64encode(imgHandler)
    except:
        print("the image could not be converted to base 64")
    #file_text = "<annotation>\n <filename>\ " converted_file.filename \ "</filename>\n <img>\ " converted_string \ "</img>\n <xAccel>\ " xvec \ "</xAccel>\n <yAccel>\ " yvec \ "</yAccel>\n <zAccel>\ " zvec \ "</zAccel>\n</annotation>"
    try:
        converted_file.write("<annotation>\n <filename>" + converted_file.name + "</filename>\n <img>" + str(converted_string) + "</img>\n <xAccel>" + xvec + "</xAccel>\n <yAccel>" + yvec + "</yAccel>\n <zAccel>" + zvec + "</zAccel>\n</annotation>")
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
importer("snowwhite.jpg", 0, 0, 0)
