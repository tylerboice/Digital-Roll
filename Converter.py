import base64

def importer(picture, xAccel, yAccel, zAccel):
    #file = open("test.xml", "w")
    image = picture
    xvec = xAccel
    yvec = yAccel
    zvec = zAccel
    convert(image, xvec, yvec, zvec)

def convert(image, xvec, yvec, zvec):
    converted_file = open("test.xml", "w")
    with open(image, "rb") as img_file:
        imgHandler = img_file.read()
    converted_string = base64.b64encode(imgHandler)
    #file_text = "<annotation>\n <filename>\ " converted_file.filename \ "</filename>\n <img>\ " converted_string \ "</img>\n <xAccel>\ " xvec \ "</xAccel>\n <yAccel>\ " yvec \ "</yAccel>\n <zAccel>\ " zvec \ "</zAccel>\n</annotation>"
    converted_file.write("<annotation>\n <filename>" + converted_file.name + "</filename>\n <img>" + str(converted_string) + "</img>\n <xAccel>" + str(xvec) + "</xAccel>\n <yAccel>" + str(yvec) + "</yAccel>\n <zAccel>" + str(zvec) + "</zAccel>\n</annotation>")
    converted_file.close()
    output(converted_file)

def output(output_file):
    return output_file

#testpic = open("snowwhite.jpg", "rb") as img_file
#file = open("test.xml", "w")
#file.close
#im = Image.open(r"/snowwhite")
importer("snowwhite.jpg", 0, 0, 0)
