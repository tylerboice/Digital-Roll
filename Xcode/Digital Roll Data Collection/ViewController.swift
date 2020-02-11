//
//  ViewController.swift
//  Digital Roll Data Collection
//
//  Created by Digital Roll on 2/10/20.
//  Copyright © 2020 Digital Roll. All rights reserved.
//

import UIKit
import AVFoundation

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // Initalize variables used for capturing the image
    let captureSession = AVCaptureSession()
    var previewLayer:CALayer!
    var captureDevice:AVCaptureDevice!
    var takePhoto = false

    // Tell the device to run the camera
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        prepareCamera()
    }
    
    // Assuming there is a capture device, start a live camrea session
    func prepareCamera() {
        captureSession.sessionPreset = AVCaptureSession.Preset.photo
        
        if let availableDevices = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: AVMediaType.video, position: .back).devices.first {
            captureDevice = availableDevices
            beginSession()
        }
    }
    
    // Attempt to capture the camrea feed
    func beginSession() {
        do {
            let captureDeviceInput = try AVCaptureDeviceInput(device: captureDevice)
            
            captureSession.addInput(captureDeviceInput)
        } catch {
            print(error.localizedDescription)
        }
        
        // Now that we know there must be a session, start the feed
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        self.view.layer.addSublayer(self.previewLayer)
        self.previewLayer.frame = self.view.layer.frame
        captureSession.startRunning()
        
        // Display the output of our recording device
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as String):NSNumber(value:kCVPixelFormatType_32BGRA)]
        
        // Ignore any dropped frames to prevent lag
        dataOutput.alwaysDiscardsLateVideoFrames = true
        
        // add the session output
        if captureSession.canAddOutput(dataOutput) {
            captureSession.addOutput(dataOutput)
        }
        
        captureSession.commitConfiguration()
        
        let queue = DispatchQueue(label: "com.digitalroll.Digital-Roll-Data-Collection.captureQueue")
        dataOutput.setSampleBufferDelegate(self, queue: queue)
    }
    
    // Sets take photo to true when camera button pressed
    @IBAction func takePhoto(_ sender: Any) {
        takePhoto = true
    }
    
    
    
    //func captureOutput(_ captureOutput: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Sets take photo to false after button is pressed, so we only take one photo
        if takePhoto {
            takePhoto = false
        
        // Get image
            if let image = self.getImageSampleBuffer(buffer: sampleBuffer) {
                let photoVC = UIStoryboard(name: "Main", bundle: nil).instantiateViewController(withIdentifier: "PhotoVC") as! PhotoViewController
                
                photoVC.takenPhoto = image
                
                DispatchQueue.main.async {
                    self.present(photoVC, animated: true, completion: {
                        self.stopCaptureSession()
                    })
                }
            }
        }
    }
    
    // Returns an image smaple buffer
    func getImageSampleBuffer(buffer:CMSampleBuffer) -> UIImage?{
        
        // Create an Image container
        if let pixelBuffer = CMSampleBufferGetImageBuffer(buffer) {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            
            // Define Image dimensions
            let imageRect = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
            
            // If an image is captured return that image
            if let image = context.createCGImage(ciImage, from: imageRect) {
                return UIImage(cgImage: image, scale: UIScreen.main.scale, orientation: .right)
            }
        }
        // If an image fails to capture return nothing
        return nil
    }
    
    func stopCaptureSession () {
        self.captureSession.stopRunning()
        
        if let inputs = captureSession.inputs as? [AVCaptureDeviceInput] {
            for input in inputs {
                self.captureSession.removeInput(input)
            }
        }
    }
}
