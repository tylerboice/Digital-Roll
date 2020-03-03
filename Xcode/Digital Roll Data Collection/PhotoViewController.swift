//
//  PhotoViewController.swift
//  Digital Roll Data Collection
//
//  Created by Digital Roll on 2/10/20.
//  Copyright © 2020 Digital Roll. All rights reserved.
//

import UIKit
import CoreMotion

class PhotoViewController: UIViewController {
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
        let file = "test.xml"
        
        // Create xml formated code
        let text = "<annotation>\n\t<filename>\(file)</filename>\n\t<img>\(img.toString())</img>\n\t<xAccel>\(xAccel)</xAccel>\n\t<yAccel>\(yAccel)</yAccel>\n\t<zAccel>\(zAccel)</zAccel>\n</annotation>"
        
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

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}

extension UIImage {
    func toString() -> String? {
        let data: Data? = self.pngData()
        return data?.base64EncodedString(options: .endLineWithLineFeed)
    }
}
