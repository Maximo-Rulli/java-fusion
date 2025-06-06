package fusion;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
    public static void main(String[] args) throws IOException {
        INDArray th = (Nd4j.create(new float[]{2})).broadcast(1,1,28,28);
        //System.out.println("Boradcasted " + t.broadcast(1,1,28,28));


        INDArray a = Nd4j.create(new float[]{
            1, 2, 3, 4, 
            5, 6, -7, -8, 
            1, 2, 3, 4, 
            5, 2, 7, 8,

            -9, -8, -7, -6, 
            -5, -4, -3, -2, 
            -9, -8, -7, -6, 
            -5, -4, -3, -2}, new int[]{2, 4, 4});

        //INDArray image = a.broadcast(3, 2, 4, 4);

        INDArray b = Nd4j.create(new float[]{-9, -8, -7, -6, -5, -4, -3, -2, -9, -8, -7, -6, -5, -4, -3, -2}, new int[]{1, 1, 4, 4});
        //INDArray result = a.add(b);
        //System.out.println("Sum:\n" + result);

        //Layers.maxPool(a, 2, 2);

        //System.out.println(Layers.concat(a, a));
        System.out.println(b);

        Layers TestL = new Layers("down1_0_block1_2", "down1_0_block1_2");

        INDArray testB = Nd4j.rand(new int[]{1, 64, 28, 28});

        TestL.Conv(testB, 1, 0);
        //Layers.Conv(b, 1, 1, 2, 1, 0);
        // Instantiate the model
        //DDPM model = new DDPM(100);

        // How to save and load matrices
        /*
        Nd4j.saveBinary(a, new File("matrix.bin"));
        System.out.println(Nd4j.readBinary(new File("matrix.bin")));
        */
        
    }
}
