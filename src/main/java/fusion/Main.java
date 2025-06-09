package fusion;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
  public static void main(String[] args) throws IOException, InterruptedException {
    INDArray th = (Nd4j.create(new float[]{2})).broadcast(1,1,28,28);
    //System.out.println("Boradcasted " + t.broadcast(1,1,28,28));

    INDArray a = Nd4j.create(new float[]{
        1, 2,
        6, 7,

        0.1f, 0.2f,
        0.6f, 0.7f,

        -1, -2,
        -6, -7,}, new int[]{1, 3, 2, 2});

    //INDArray image = a.broadcast(3, 2, 4, 4);

    INDArray k = Nd4j.create(new float[]{
                            1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,
                            1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,}, new int[]{2, 3, 2, 2});
    
    INDArray b = Nd4j.create(new float[]{100,1}, new int[]{2});
    //Layers.maxPool(a, 2, 2);

    //System.out.println(Layers.concat(a, a));

    Layer TestL = new Layer(k, b);

    INDArray testB = Nd4j.rand(new int[]{1, 2, 4, 4});

    System.out.println(TestL.TranspConv(a, 2, 0));
    
    Correctness.run();
    //Layers.Conv(b, 1, 1, 2, 1, 0);
    
  }
}
