package fusion;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
  public static void main(String[] args) throws IOException, InterruptedException {
    INDArray th = (Nd4j.create(new float[]{2})).broadcast(1,1,28,28);
    //System.out.println("Boradcasted " + t.broadcast(1,1,28,28));

    INDArray a = Nd4j.create(new float[]{
        1, 2, 3, 4, 5}, new int[]{1, 5});

    //INDArray image = a.broadcast(3, 2, 4, 4);

    INDArray k = Nd4j.create(new float[]{
                            1, 0, 0, 0, 0,
                            0, 1, 0, 0, 0,
                            0, 0, 1, 0, 0,
                            0, 0, 0, 1, 0,
                            0, 0, 0, 0, 1,
                            1, 0, 1, 0, 1,
                          
                          }, new int[]{6,5});
    
    INDArray b = Nd4j.create(new float[]{0,0,1,0,0,-9}, new int[]{6});
    //Layers.maxPool(a, 2, 2);

    //System.out.println(Layers.concat(a, a));

    Layer TestL = new Layer(k, b);

    INDArray testB = Nd4j.rand(new int[]{1, 2, 4, 4});

    System.out.println(TestL.Linear(a, true));
    
    Correctness.run();
    //Layers.Conv(b, 1, 1, 2, 1, 0);
    
  }
}
