import java.util.ArrayList;
import java.util.Random;
import java.io.*;
import java.text.DecimalFormat;
import java.lang.Math;
import java.util.List;


public class Mlp {

	// main constructor
	public Mlp(int nn_neurons[])
	{
		Random rand = new Random();

		// create the required layers
		_layers = new ArrayList<Layer>();
		for (int i = 0; i < nn_neurons.length; ++i)
			_layers.add(
					new Layer(
							i == 0 ?
							0 : nn_neurons[i - 1],
							nn_neurons[i], rand)
					);

		_delta_w = new ArrayList<float[][]>();
		for (int i = 1; i < nn_neurons.length; ++i)
			_delta_w.add(new float
						[_layers.get(i).size()]
						[_layers.get(i).getWeights(0).length]
					 );

		_grad_ex = new ArrayList<float[]>();
		for (int i =  0; i < nn_neurons.length; ++i)
			_grad_ex.add(new float[_layers.get(i).size()]);
	}

	public float[] evaluate(float[] inputs)
	{
		// propagate the inputs through all neural network
		// and return the outputs
		assert(false);

		float outputs[] = new float[inputs.length];

		for( int i = 0; i < _layers.size(); ++i ) {
			outputs = _layers.get(i).evaluate(inputs);
			inputs = outputs;
		}

		return outputs;
	}

	private float evaluateError(float nn_output[], float desired_output[])
	{
		float d[];

		// add bias to input if necessary
		if (desired_output.length != nn_output.length)
			d = Layer.add_bias(desired_output);
		else
			d = desired_output;

		assert(nn_output.length == d.length);

		float e = 0;
		for (int i = 0; i < nn_output.length; ++i)
			e += (nn_output[i] - d[i]) * (nn_output[i] - d[i]);

		return e;
	}

	public float evaluateQuadraticError(ArrayList<float[]> examples,
								   ArrayList<float[]> results)
	{
		// this function calculate the quadratic error for the given
		// examples/results sets
		assert(false);

		float e = 0;

		for (int i = 0; i < examples.size(); ++i) {
			e += evaluateError(evaluate(examples.get(i)), results.get(i));
			//System.out.println("....." + e);
		}

		return e;
	}

	private void evaluateGradients(float[] results)
	{
		// for each neuron in each layer
		for (int c = _layers.size()-1; c >= 0; --c) {
			for (int i = 0; i < _layers.get(c).size(); ++i) {
				// if it's output layer neuron
				if (c == _layers.size()-1) {
					_grad_ex.get(c)[i] =
						2 * (_layers.get(c).getOutput(i) - results[0])
						  * _layers.get(c).getActivationDerivative(i);
				}
				else { // if it's neuron of the previous layers
					float sum = 0;
					for (int k = 1; k < _layers.get(c+1).size(); ++k)
						sum += _layers.get(c+1).getWeight(k, i) * _grad_ex.get(c+1)[k];
					_grad_ex.get(c)[i] = _layers.get(c).getActivationDerivative(i) * sum;
				}
			}
		}
	}

	private void resetWeightsDelta()
	{
		// reset delta values for each weight
		for (int c = 1; c < _layers.size(); ++c) {
			for (int i = 0; i < _layers.get(c).size(); ++i) {
				float weights[] = _layers.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j){
					//System.out.println("c: " + String.valueOf(c) + ", i: " + String.valueOf(i) + "j: " + String.valueOf(j));
					_delta_w.get(c-1)[i][j] = 0;
				}
	    }
		}
	}

	private void evaluateWeightsDelta()
	{
		// evaluate delta values for each weight
		for (int c = 1; c < _layers.size(); ++c) {
			for (int i = 0; i < _layers.get(c).size(); ++i) {
				float weights[] = _layers.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j)
					_delta_w.get(c-1)[i][j] += _grad_ex.get(c)[i]
					     * _layers.get(c-1).getOutput(j);
			}
		}
	}

	private void updateWeights(float learning_rate)
	{
		for (int c = 1; c < _layers.size(); ++c) {
			for (int i = 0; i < _layers.get(c).size(); ++i) {
				float weights[] = _layers.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j)
					_layers.get(c).setWeight(i, j, _layers.get(c).getWeight(i, j)
							- (learning_rate * _delta_w.get(c-1)[i][j]));
	        }
		}
	}

	private void batchBackPropagation(ArrayList<float[]> examples,
									  ArrayList<float[]> results,
									  float learning_rate)
	{
		resetWeightsDelta();

		for (int l = 0; l < examples.size(); ++l) {
			evaluate(examples.get(l));
			evaluateGradients(results.get(l));
			evaluateWeightsDelta();
		}

		updateWeights(learning_rate);
	}

	public void learn(ArrayList<float[]> examples,
					  ArrayList<float[]> results,
					  float learning_rate)
	{
		// this function implements a batched back propagation algorithm
		assert(false);

		float e = Float.POSITIVE_INFINITY;

		while (e > 0.001f) {
			System.out.println("entre " +  e);
			batchBackPropagation(examples, results, learning_rate);

			e = evaluateQuadraticError(examples, results);
		}
		System.out.println("sali " +  e);
	}

	private ArrayList<Layer> _layers;
	private ArrayList<float[][]> _delta_w;
	private ArrayList<float[]> _grad_ex;


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		/*
		// initialization
		ArrayList<float[]> ex = new ArrayList<float[]>();
		ArrayList<float[]> out = new ArrayList<float[]>();
		for (int i = 0; i < 4; ++i) {
			ex.add(new float[2]);
			out.add(new float[1]);
		}

		// fill the examples database
		ex.get(0)[0] = -1; ex.get(0)[1] = 1;  out.get(0)[0] = 1;
		ex.get(1)[0] = 1;  ex.get(1)[1] = 1;  out.get(1)[0] = -1;
		ex.get(2)[0] = 1;  ex.get(2)[1] = -1; out.get(2)[0] = 1;
		ex.get(3)[0] = -1; ex.get(3)[1] = -1; out.get(3)[0] = -1;
		*/

		int nn_neurons[] = {
				2,//ex.get(0).length, 	// layer 1: input layer - 2 neurons
				6, 	// layer 2: hidden layer - 2n + 1 neurons = 21 neurons
				1			// layer 3: output layer - 1 neuron
		};

		Mlp mlp = new Mlp(nn_neurons);

		/**/
		int cantidadVectores = 5;
		Random aleatorio = new Random(System.currentTimeMillis());
    DecimalFormat df = new DecimalFormat("#.########");

    //GENERAR Y GUARDAR 1500 VECTORES ALEATORIOS Y OBTENER EL RESULTADO DE LA FUNCION
    ArrayList<float[]> ex = new ArrayList<float[]>();
		ArrayList<float[]> out = new ArrayList<float[]>();
    for (int i = 0; i < cantidadVectores; ++i) {
      float[] x = new float[10];
      for(int j = 0; j < 10; j++){
        float num = aleatorio.nextFloat();
        float result = -5 + (10 * num);
        String d = df.format(result);
        //System.out.print(d + "; ");
        x[j] = Float.parseFloat(d.replace(",","."));
      }
			ex.add(x);
      //System.out.println(x[0] + "; " + x[1]);
      //System.out.println(" ");
			out.add(mlp.rastrigin(x));
		}
		/**/

		try {
			PrintWriter fout = new PrintWriter(new FileWriter("plot.dat"));
			fout.println("#\tX\tY");

			for (int i = 0; i < 5; ++i) {
				mlp.learn(ex, out, 0.3f);
				float error = mlp.evaluateQuadraticError(ex, out);
				System.out.println(i + " -> error : " + error);
				fout.println("\t" + i + "\t" + error);
			}

			fout.close();
		} catch (IOException e){
			e.printStackTrace();
		}
	}

	public float[] rastrigin (float[] x){
    float res[] = new float[1];

    //z = x - o; x vector entrada; o vector solucion optimo
    float[] z = new float[10];
    float[] o = {3.84659436f, 4.323622057f,
                          -2.82162937f, 0.646538176f,
                          4.338201865f, 2.440869972f,
                          -3.459310241f, 2.337058759f,
                          3.979980558f, 4.290014324f};
    z[0] = x[0] - o[0];    z[1] = x[1] - o[1];
    z[2] = x[2] - o[2];    z[3] = x[3] - o[3];
    z[4] = x[4] - o[4];    z[5] = x[5] - o[5];
    z[6] = x[6] - o[6];    z[7] = x[7] - o[7];
    z[8] = x[8] - o[8];    z[9] = x[9] - o[9];

    float sumatoria = 0;
    for (int i = 0; i < 10 ; i++){
          float aux = (float) (2.0f * Math.PI * z[i]);
          sumatoria = (float) (sumatoria + ((z[i] * z[i])
                                      - (10.0f * Math.cos(aux)) + 10.0f));
    }
    res[0] = sumatoria - 330;
    System.out.println("Res: " + res[0]);
    return res;
  }
}
