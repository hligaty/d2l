package io.github.hligaty.ssd;

import ai.djl.basicdataset.cv.ObjectDetectionDataset;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.translate.Pipeline;
import ai.djl.util.PairList;
import ai.djl.util.Progress;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import com.fasterxml.jackson.dataformat.xml.annotation.JacksonXmlElementWrapper;
import com.fasterxml.jackson.dataformat.xml.annotation.JacksonXmlProperty;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

/**
 * 读取 VOC 标注文件
 */
public class VocDetection extends ObjectDetectionDataset {
    private final List<Path> imagePaths;
    private final List<PairList<Long, Rectangle>> labels;
    private final Map<String, Long> classes = new LinkedHashMap<>();

    private final String repositoryDir;
    private boolean prepared;

    public VocDetection(Builder builder) {
        super(builder);
        repositoryDir = builder.repositoryDir;
        imagePaths = new ArrayList<>();
        labels = new ArrayList<>();
    }

    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }
        Path root = Paths.get(repositoryDir);
        Path classesPath = root.resolve("classes.txt");
        Files.readAllLines(classesPath).forEach(s -> classes.put(s, (long) classes.size()));
        try (Stream<Path> stream = Files.walk(root, 1)) {
            XmlMapper xmlMapper = XmlMapper.builder()
                    .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
                    .build();
            stream
                    .filter(path -> path.toString().endsWith("xml"))
                    .forEach(path -> {
                        try {
                            VocAnnotation vocAnnotation = xmlMapper.readValue(new String(Files.readAllBytes(path)), VocAnnotation.class);
                            imagePaths.add(Paths.get(vocAnnotation.path));
                            VocAnnotation.Size size = vocAnnotation.size;
                            PairList<Long, Rectangle> list = new PairList<>();
                            for (VocAnnotation.Obj object : vocAnnotation.objects) {
                                VocAnnotation.Obj.Bndbox bndbox = object.bndbox;
                                Rectangle rectangle = new Rectangle(new Point(
                                        bndbox.xmin / size.width,
                                        bndbox.ymin / size.height),
                                        bndbox.xmax / size.width,
                                        bndbox.ymax / size.height);
                                list.add(classes.get(object.name), rectangle);
                            }
                            labels.add(list);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    });
        }
        prepared = true;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public PairList<Long, Rectangle> getObjects(long index) {
        return labels.get((int) index);
    }

    @Override
    public List<String> getClasses() {
        if (prepared) {
            return new ArrayList<>(classes.keySet());
        }
        try {
            return Files.readAllLines(Paths.get(repositoryDir).resolve("classes.txt"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected Image getImage(long index) throws IOException {
        int idx = Math.toIntExact(index);
        return ImageFactory.getInstance().fromFile(imagePaths.get(idx));
    }

    @Override
    public Optional<Integer> getImageWidth() {
        return Optional.empty();
    }

    @Override
    public Optional<Integer> getImageHeight() {
        return Optional.empty();
    }

    @Override
    protected long availableSize() {
        return imagePaths.size();
    }

    public static final class Builder extends BaseBuilder<Builder> {
        String repositoryDir;

        public Builder optRepositoryDir(String repositoryDir) {
            this.repositoryDir = repositoryDir;
            return self();
        }

        @Override
        protected Builder self() {
            return this;
        }

        public VocDetection build() {
            if (pipeline == null) {
                pipeline = new Pipeline(new ToTensor());
            }
            return new VocDetection(this);
        }
    }

    static class VocAnnotation {
        public String path;
        public Size size;
        @JacksonXmlProperty(localName = "object")
        @JacksonXmlElementWrapper(useWrapping = false)
        public List<Obj> objects;


        static class Size {
            public int width;
            public int height;
        }

        static class Obj {
            public String name;
            public Bndbox bndbox;

            static class Bndbox {
                public double xmin;
                public double ymin;
                public double xmax;
                public double ymax;
            }
        }
    }
}
