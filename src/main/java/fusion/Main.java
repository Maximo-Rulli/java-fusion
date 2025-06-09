package fusion;

import java.io.IOException;

public class Main {
  public static void main(String[] args) throws IOException, InterruptedException {
    UNet Model = new UNet();
    Correctness CorrectnessTests = new Correctness(Model);
    CorrectnessTests.run();
    Tests.run();

    DDPM JavaFusion = new DDPM(Model, 1500);
    
    ImageSaver.saveImage(JavaFusion.sample(100), "mnist.png");
  }
}
