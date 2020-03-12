//
//  PhotoViewController.swift
//  Digital Roll Data Collection
//
//  Created by Digital Roll on 2/10/20.
//  Copyright © 2020 Digital Roll. All rights reserved.
//

import UIKit
import CoreMotion

class PhotoViewController: UIViewController, UIPickerViewDataSource, UIPickerViewDelegate {
    // Initalize File
    var returnFile: URL?
    
    // Initalize photo window
    var takenPhoto:UIImage?
    @IBOutlet weak var imageView: UIImageView!
    
    // Initalize Accelerometer readout
    @IBOutlet weak var xAccel: UITextField!
    @IBOutlet weak var yAccel: UITextField!
    @IBOutlet weak var zAccel: UITextField!
    var motion = CMMotionManager()
    
    // Initalize Picker Views
    @IBOutlet weak var diceShapePV: UIPickerView!
    let diceShape = ["d4", "d6", "d8", "d10", "d%", "d12", "d20"]
    var diceShapeValue: String = ""
        
    @IBOutlet weak var diceNumberPV: UIPickerView!
    let diceNumber = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    var diceNumberValue: String = ""
    
    // Initalize touch event values
    var touchHandler = UIEvent()
    var touchSet = Set<UITouch>()
    
    // Intalize touch corners for rectangle
    var touchArray: [CGFloat] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()

        if let availableImage = takenPhoto {
            imageView.image = availableImage
        }
        
        readAccelerometer()
    }
    
    // Back button
    @IBAction func goBack(_ sender: Any) {
        self.dismiss(animated: true, completion: nil)
    }
    
    // Checks accelerometer data at moment photo is taken
    func readAccelerometer() {
        motion.accelerometerUpdateInterval = TimeInterval(Int(UInt8.max))
        motion.startAccelerometerUpdates(to: OperationQueue.current!) {(data, error) in print(data as Any)
            self.motion.stopAccelerometerUpdates()
          
            if let trueData = data {
                self.view.reloadInputViews()
                let x = trueData.acceleration.x
                let y = trueData.acceleration.y
                let z = trueData.acceleration.z
              
                self.xAccel.text = "X: \(Double(x).rounded(toPlaces: 3))"
                self.yAccel.text = "Y: \(Double(y).rounded(toPlaces: 3))"
                self.zAccel.text = "Z: \(Double(z).rounded(toPlaces: 3))"
            }
        }
    }
    
    // Share button functionality
    @IBAction func sharePressed(_ sender: Any) {
        // Dropbox only works with 1 arguement therefore pass just xml file
        convertXML(img: takenPhoto!, xAccel: self.xAccel.text!, yAccel: self.yAccel.text!, zAccel: self.zAccel.text!)
        
        let activityVC = UIActivityViewController(activityItems: [returnFile as Any], applicationActivities: nil)
        activityVC.popoverPresentationController?.sourceView = self.view
        
        self.present(activityVC, animated: true, completion: nil)
        
    }
    
    func convertXML(img: UIImage, xAccel: String, yAccel: String, zAccel: String){
        // Create file name
        let file = diceShapeValue + "-" + diceNumberValue + ".xml"
        
        // Get Image stats
        let heightInPixels = takenPhoto!.size.height * takenPhoto!.scale
        let widthInPixels = takenPhoto!.size.width * takenPhoto!.scale
        
        // Create xml formated code
        let text = "<annotation>\n\t<filename>\(file)</filename>\n\t<size>\n\t\t<width>\(widthInPixels)</width>\n\t\t<height>\(heightInPixels)</height>\n\t\t<depth>3</depth\n\t<size>\n\n\t<xAccel>\(xAccel)</xAccel>\n\t<yAccel>\(yAccel)</yAccel>\n\t<zAccel>\(zAccel)</zAccel>\n\n\t<object>\n\t\t<name>\(diceShapeValue)-\(diceNumberValue)</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<xmin>\(touchArray[3])</xmin>\n\t\t\t<ymin>\(touchArray[2])</ymin>\n\t\t\t<xmax>\(touchArray[0])</xmax>\n\t\t\t<ymax>\(touchArray[1])</ymax>\n\t\t</bndbox>\n\t</object>\n\n\t<img>\(img.toString())</img>\n</annotation>"
        
        // Write file in phone directory
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = dir.appendingPathComponent(file)

            // Write the XML if possible
            do {
                try text.write(to: fileURL, atomically: false, encoding: .utf8)
            }
            catch {}
            returnFile = fileURL
        }
    }
    
    // Function to decide number of objects to return form pickerview
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    // Function to return value in picker view
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        if (pickerView == diceShapePV){
            diceShapeValue = diceShape[row]
            return diceShape[row]
        }
        if (pickerView == diceNumberPV){
            diceNumberValue = diceNumber[row]
            return diceNumber[row]
        }
        return "Error"
    }
    
    // Determines how many numbers for die face are shwon
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        if pickerView == diceShapePV {
            return diceShape.count
        }
        if pickerView == diceNumberPV {
            return diceNumber.count
        }
        return 0
    }
    
    // bounding Box button funcitonality
    @IBAction func boundingBoxPressed(_ sender: Any) {
        touchesBegan(touchSet, with: touchHandler)
        print(touchArray)
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let touch = touches.first {
            let position = touch.location(in: view)
            touchArray.append(position.x)
            touchArray.append(position.y)
        }
        touchesEnded(touchSet, with: touchHandler)
    }
    
}

extension UIImage {
    func toString() -> String? {
        let data: Data? = self.pngData()
        return data?.base64EncodedString(options: .endLineWithLineFeed)
    }
}
