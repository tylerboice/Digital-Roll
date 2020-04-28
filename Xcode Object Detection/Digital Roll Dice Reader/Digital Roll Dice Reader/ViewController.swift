//
//  ViewController.swift
//  Digital Roll Dice Reader
//
//  Created by Digital Roll on 4/9/20.
//  Copyright © 2020 Digital Roll. All rights reserved.
//

import UIKit
import AVKit
import Vision
import CoreML
import AVFoundation

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    // Create variables for objects in view
    @IBOutlet weak var innerView: UIView!
    @IBOutlet weak var viewLabel: UILabel!
    @IBOutlet weak var titleLabel: UILabel!
    
    var previewLayer: AVCaptureVideoPreviewLayer?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        updateLabel(newLabel: "Waiting for an object to be detected.")
        
        // Start recording
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        // back camera video capture
        guard let captureDevice = AVCaptureDevice.default(for: .video)
            else {self.quickErr(myLine: #line, inputStr: ""); return}
        
        try? captureDevice.lockForConfiguration()
        captureDevice.activeVideoMaxFrameDuration = CMTime(value: 5, timescale: 120)
        captureDevice.activeVideoMinFrameDuration = CMTime(value: 5, timescale: 120)
        captureDevice.unlockForConfiguration()
        
        
        guard let input = try? AVCaptureDeviceInput(device: captureDevice)
            else{self.quickErr(myLine: #line, inputStr: ""); return}
        captureSession.addInput(input)
        captureSession.startRunning()
        
        self.previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        self.previewLayer?.frame.size = self.innerView.frame.size
        self.previewLayer?.videoGravity = AVLayerVideoGravity.resizeAspectFill
        self.innerView.layer.addSublayer(self.previewLayer!)
        self.previewLayer?.frame = view.frame
        
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "VideoQueue"))
        captureSession.addOutput(dataOutput)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        self.previewLayer?.frame.size = self.innerView.frame.size
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection){
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {self.quickErr(myLine: #line, inputStr: ""); return}
        guard let model = try? VNCoreMLModel(for: core_ml().model) else {self.quickErr(myLine: #line, inputStr: ""); return}
        let request = VNCoreMLRequest(model: model) {(finishedReq, err) in
            guard let results = finishedReq.results as? [VNClassificationObservation] else {return}
            guard let firstObservation = results.first else {return}
            
            var myMessage = ""
            var myConfidence = 0
            
            if (firstObservation.confidence > 0) {
                myConfidence = Int(firstObservation.confidence * 100)
                let myIdentifier = firstObservation.identifier.split(separator: ",")
                myMessage = "The model is \(myConfidence)% confident this object is: \(myIdentifier[0])."
            }
            else {
                myMessage = "The model is not confident enough to classify the object"
            }
            
            self.updateLabel(newLabel: myMessage)
        }
        
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }

    // Updates the values in the labels
    func updateLabel(newLabel: String) {
        DispatchQueue.main.async {
            self.viewLabel?.text = newLabel
        }
    }
    
    func quickErr(myLine: Int, inputStr: String = ""){
        print("===> Guard Error \(inputStr) : \n file: \(#file)\n line: \(myLine)\n function: \(#function)")
    }
}

