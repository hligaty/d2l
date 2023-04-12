package org.example;

import ai.djl.MalformedModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import org.junit.jupiter.api.Test;

import java.io.IOException;

public class ModelTest {

    @Test
    public void run() throws TranslateException, ModelNotFoundException, MalformedModelException, IOException {
        io.github.hligaty.demo.det.pedestrian.Inference.run("src/test/resources/pedestrain.png");  // 行人检测 PaddleDetection 模型
        io.github.hligaty.demo.det.seal.Inference.run("src/test/resources/seal-det.jpeg");  // 印章检测 PaddleDetection 模型
        io.github.hligaty.demo.det.yolov5.Inference.run("src/test/resources/cat-dog.png");    // 目标检测 PyTorch 模型
        io.github.hligaty.demo.ocr.det.Inference.run("src/test/resources/ocr_det.png");    // OCR 文字检测
        io.github.hligaty.demo.ocr.rec.Inference.run("src/test/resources/ocr_rec.png");    // OCR 文字识别
        io.github.hligaty.demo.det.vehicle.Inference.run("src/test/resources/pedestrain.png"); // 车辆识别
    }
}
