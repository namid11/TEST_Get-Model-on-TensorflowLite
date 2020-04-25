//
//  ViewController.swift
//  TEST_Get-Model-on-TensorflowLite
//
//  Created by NH on 2020/04/25.
//  Copyright © 2020 NH. All rights reserved.
//

import UIKit
import TensorFlowLite

class ViewController: UIViewController {

    var interpreter: Interpreter? = nil
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    @IBAction func loadButton(_ sender: Any) {
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
          forResource: "converted_model",
          ofType: "tflite"
        ) else {
          print("Failed to load the model file with name: ~")
          return
        }
        
        do {
          // Create the `Interpreter`.
          interpreter = try Interpreter(modelPath: modelPath)
          // Allocate memory for the model's input `Tensor`s.
          // メモリ確保。これはモデル読み込み直後に必須
          try interpreter?.allocateTensors()
        } catch let error {
          print("Failed to create the interpreter with error: \(error.localizedDescription)")
          return
        }
        
        guard let intp = interpreter else {
            return
        }
        
        do {
            let inputData = createXData()
            try intp.copy(inputData, toInputAt: 0)
            try intp.invoke()
            // Get the output `Tensor` to process the inference results.
            let outputTensor = try intp.output(at: 0)
            print(outputTensor)
            let outputValues = decodeData(output: outputTensor.data)
            print(outputValues)
            dispSinWave(X: createXLine(), Y: outputValues)
        } catch let error {
            print("error:", error)
        }
        
    }
    
    
    private func createXData() -> Data {
        var bytes: [UInt8] = []
        let xLine = createXLine()
        for var value in xLine {
            let valueBytes = Data(bytes: &value, count: (Float32.significandBitCount + Float32.exponentBitCount + 1) / 8)
            let bytesArray = Array<UInt8>(valueBytes)
            bytes += bytesArray
        }
        
        return Data(bytes: bytes, count: bytes.count)
    }
    
    private func decodeData(output: Data) -> [Float32] {
        var decodeValue: [Float32] = []
        let float32Bytes = (Float32.significandBitCount + Float32.exponentBitCount + 1) / 8
        for i in 0 ..< output.count / float32Bytes {
            let data = output[i*float32Bytes..<(i+1)*float32Bytes]
            let value = data.withUnsafeBytes { $0.load(as: Float32.self) }
            decodeValue.append(value)
        }
        
        return decodeValue
    }
    
    private func createXLine() -> [Float32] {
        var array: [Float32] = []
        let NUM = 1000
        let delta = 2*Float.pi / Float(NUM)
        for i in 0 ..< NUM {
            let value = Float32(i)*Float32(delta)
            array.append(value)
        }
        return array
    }
    
    private func dispSinWave(X: [Float32], Y: [Float32]) {
        let layer = CAShapeLayer()
        layer.frame = self.view.layer.bounds
        let path = UIBezierPath()
        for i in 0..<X.count {
            let xBias = (layer.frame.width / CGFloat(X.count)) * CGFloat(i)
            let yBias = layer.frame.height / 2.0
            let yFactor: CGFloat = yBias / 2.0
            let x = CGFloat(X[i]) + xBias
            let y = CGFloat(-Y[i])*yFactor + yBias  // 表示する時はy軸は逆になります
            if i == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
        layer.path = path.cgPath
        layer.lineWidth = 5.0
        layer.strokeColor = UIColor.cyan.cgColor
        self.view.layer.addSublayer(layer)
    }
}

