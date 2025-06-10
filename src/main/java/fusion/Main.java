package fusion;

import java.io.IOException;

public class Main {
  public static void main(String[] args) throws IOException, InterruptedException {
    final int NUM_SAMPLES = 5;
    UNet Model = new UNet();
    Correctness CorrectnessTests = new Correctness(Model);
    CorrectnessTests.run();
    //Tests.run();

    //DDPM JavaFusion = new DDPM(Model, 1500);

    for (int i = 0; i < NUM_SAMPLES; i++) {
        int sampleIndex = i;
        Runnable task = () -> {
            // Create a separate DDPM instance per thread (just in case)
            DDPM JavaFusion = new DDPM(Model, 1500);
            JavaFusion.sample(100, sampleIndex);
        };

        Thread thread = new Thread(task);
        thread.start();
    }
    
    //ImageSaver.saveImage(JavaFusion.sample(100), "mnist.png");
  }
}
