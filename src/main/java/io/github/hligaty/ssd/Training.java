package io.github.hligaty.ssd;

import ai.djl.Model;
import ai.djl.basicdataset.cv.ObjectDetectionDataset;
import ai.djl.basicdataset.cv.PikachuDetection;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.BoundingBoxError;
import ai.djl.training.evaluator.SingleShotDetectionAccuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SingleShotDetectionLoss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;

import static io.github.hligaty.ssd.Models.IMAGE_SIZE;

public class Training {

    public static final int BATCH_SIZE = 1;
    public static final int NUM_EPOCH = 3;

    public static void main(String[] args) throws IOException, TranslateException {
        ObjectDetectionDataset dataset = getVocDataset();
        RandomAccessDataset[] vocDataset = dataset.randomSplit(9, 1);
        RandomAccessDataset trainingSet = getVocDataset();
        RandomAccessDataset testSet = vocDataset[1];

        DefaultTrainingConfig config = setupTrainingConfig();

        try (Model model = Models.getModel();
             Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());

            Shape inputShape = new Shape(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE);
            trainer.initialize(inputShape);

            EasyTrain.fit(trainer, NUM_EPOCH, trainingSet, testSet);

            Models.saveSynset(Models.MODEL_PATH, dataset.getClasses());
        }
    }

    private static ObjectDetectionDataset getDataset(Dataset.Usage usage, int limit) throws IOException {
        PikachuDetection pikachuDetection = PikachuDetection.builder()
                .optUsage(usage)
                .optLimit(limit)
                .addTransform(new ToTensor())
                .setSampling(BATCH_SIZE, false)
                .build();
        pikachuDetection.prepare(new ProgressBar());
        return pikachuDetection;
    }

    public static ObjectDetectionDataset getVocDataset() throws IOException {
        VocDetection vocDetection = VocDetection.builder()
                .optRepositoryDir("D:\\Repository\\dataset\\my_coco\\\\annotations")
                .addTransform(new ToTensor())
                .setSampling(BATCH_SIZE, false)
                .build();
        vocDetection.prepare(new ProgressBar());
        return vocDetection;
    }

    static DefaultTrainingConfig setupTrainingConfig() {
        String accuracyName = "classAccuracy";
        String boundingBoxErrorName = "BoundingBoxError";
        SaveModelTrainingListener saveModelTrainingListener = new SaveModelTrainingListener(Models.MODEL_PATH.toString());
        saveModelTrainingListener.setSaveModelCallback(trainer -> {
            Model model = trainer.getModel();
            TrainingResult result = trainer.getTrainingResult();
            model.setProperty(accuracyName, String.format("%.5f", result.getValidateEvaluation(accuracyName)));
            model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
        });
        SingleShotDetectionLoss loss = new SingleShotDetectionLoss();
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new SingleShotDetectionAccuracy(accuracyName))
                .addEvaluator(new BoundingBoxError(boundingBoxErrorName))
                .addTrainingListeners(TrainingListener.Defaults.logging())
                .addTrainingListeners(saveModelTrainingListener);
    }
}
