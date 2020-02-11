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

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
