using System;
using TensorFlow;
//using Mono.Options;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UIKit;

namespace StoryoClassify
{
	
	public partial class ViewController : UIViewController
	{
		static string dir, modelFile, labelsFile;

		protected ViewController (IntPtr handle) : base (handle)
		{
			
			// Note: this .ctor should not contain any initialization logic.
			//ObjCRuntime.LinkWithAttribute.
		}

		public override void ViewDidLoad ()
		{
			base.ViewDidLoad ();
			//var files = options.Parse (args);
			if (dir == null) {
				dir = "/tmp";
				//Error ("Must specify a directory with -m to store the training data");
			}

			//if (files == null || files.Count == 0)
			//	Error ("No files were specified");

			//if (files.Count == 0)
			var files = new List<string> () { /*"Images/ship.jpeg",*/ "Images/cat.jpeg" };

			ModelFiles (dir);

			// Construct an in-memory graph from the serialized form.
			var graph = new TFGraph ();
			// Load the serialized GraphDef from a file.
			var model = File.ReadAllBytes (modelFile);

			graph.Import (model, "");
			using (var session = new TFSession (graph)) {
				var labels = File.ReadAllLines (labelsFile);

				foreach (var file in files) {

					const int wanted_width = 299;
					const int wanted_height = 299;
					const int wanted_channels = 3;
					const float input_mean = 0.0f;
					const float input_std = 255.0f;

					// Run inference on the image files
					// For multiple images, session.Run() can be called in a loop (and
					// concurrently). Alternatively, images can be batched since the model
					// accepts batches of image data as input.
					var tensor = CreateTensorFromImageFile (file, wanted_width, wanted_height, wanted_channels, input_mean, input_std);

					var runner = session.GetRunner ();
					runner.AddInput (graph ["input"] [0], tensor).Fetch (graph ["InceptionV3/Predictions/Reshape_1"] [0]);
					var output = runner.Run ();
					// output[0].Value() is a vector containing probabilities of
					// labels for each image in the "batch". The batch size was 1.
					// Find the most probably label index.

					var result = output [0];
					var rshape = result.Shape;
					if (result.NumDims != 2 || rshape [0] != 1) {
						var shape = "";
						foreach (var d in rshape) {
							shape += $"{d} ";
						}
						shape = shape.Trim ();
						Console.WriteLine ($"Error: expected to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape [{shape}]");
						Environment.Exit (1);
					}

					// You can get the data in two ways, as a multi-dimensional array, or arrays of arrays, 
					// code can be nicer to read with one or the other, pick it based on how you want to process
					// it
					bool jagged = true;

					var list = new List<Tuple<int, float>> ();
					var bestIdx = 0;
					float p = 0, best = 0;
					const float kThreshold = 0.1f;
					const int maxCount = 2;

					if (jagged) {
						var probabilities = ((float [] [])result.GetValue (jagged: true)) [0];
						for (int i = 0; i < probabilities.Length; i++) {
							if(list.Count >= maxCount) {
								break;
							}

							if (probabilities [i] > kThreshold) {
								list.Add (new Tuple<int, float> (i, probabilities[i]));
								bestIdx = i;
								best = probabilities [i];
							}
						}

					} else {
						var val = (float [,])result.GetValue (jagged: false);

						// Result is [1,N], flatten array
						for (int i = 0; i < val.GetLength (1); i++) {
							if (list.Count >= maxCount) {
								break;
							}

							if (val [0, i] > kThreshold) {
								list.Add (new Tuple<int, float> (i, val [0, i]));
								bestIdx = i;
								best = val [0, i];
							}
						}
					}
					foreach(Tuple<int, float> t in list)
					{	
						Console.WriteLine ($"{file} best match: [{t.Item1}] {t.Item2 * 100.0}% {labels [t.Item1]}");
					}
				}
			}
		}

		public static byte [] GetBytes (uint value)
		{
			return new byte [4] {
		    (byte)(value & 0xFF),
		    (byte)((value >> 8) & 0xFF),
		    (byte)((value >> 16) & 0xFF),
		    (byte)((value >> 24) & 0xFF) };
		}

		public static unsafe byte [] GetBytes (float value)
		{
			uint val = *((uint*)&value);
			return GetBytes (val);
		}

		// Convert the image in filename to a Tensor suitable as input to the Inception model.
		static TFTensor CreateTensorFromImageFile (string file, int wanted_width, int wanted_height, int wanted_channels, float input_mean, float input_std)
		{
			unsafe
			{
				int image_width;
				int image_height;
				int image_channels;
				// Get a byte[]
				var image_data = IOS_IMAGE_LOAD_EXTENSIONS.LoadImageFromFile (file, out image_width, out image_height, out image_channels);

				var image_tensor = new TFTensor (TFDataType.Float, new long [] { 1, wanted_width, wanted_height, wanted_channels }, wanted_width * wanted_height * wanted_channels * sizeof(float));

				//Get a pointer to the start
				fixed (byte* in_image_data = image_data) {
					//Pointer to end
					byte* in_end_image_data = (in_image_data +(image_height * image_width * image_channels));
					var vb = *in_image_data;
					//Pointer to tensor data
					float* out_image_data = (float*)image_tensor.Data.ToPointer ();

					for (int y = 0; y < wanted_height; ++y) {
						int in_y = (y * image_height) / wanted_height;
						byte* in_row = in_image_data + (in_y * image_width * image_channels);

						float* out_row = out_image_data +(y * wanted_width * wanted_channels);
						for (int x = 0; x < wanted_width; ++x) {
							int in_x = (x * image_width) / wanted_width;
							byte* in_pixel = in_row + (in_x * image_channels);
							var pp = *in_pixel;
							float* out_pixel = out_row + (x * wanted_channels);
							float val = *out_pixel;
							float hj = (float)*out_pixel;
							for (int c = 0; c < wanted_channels; ++c) {
								out_pixel [c] = (in_pixel [c] - input_mean) / input_std;
								var j = out_pixel [c];
							}
						}
					}
				}
				return image_tensor;
			}
		}

		// The inception model takes as input the image described by a Tensor in a very
		// specific normalized format (a particular image size, shape of the input tensor,
		// normalized pixel values etc.).
		//
		// This function constructs a graph of TensorFlow operations which takes as
		// input a JPEG-encoded string and returns a tensor suitable as input to the
		// inception model.
		/*static void ConstructGraphToNormalizeImage (out TFGraph graph, out TFOutput input, out TFOutput output)
		{
			// Some constants specific to the pre-trained model at:
			// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
			//
			// - The model was trained after with images scaled to 224x224 pixels.
			// - The colors, represented as R, G, B in 1-byte each were converted to
			//   float using (value - Mean)/Scale.

			const int W = 299;
			const int H = 299;
			const float Mean = 0;
			const float Scale = 255;

			graph = new TFGraph ();
			input = graph.Placeholder (TFDataType.String, null);

			var tf = ConstructGraphToNormalizeImage ();

			output = graph.Div (
				x: graph.Sub (
					x: graph.ResizeBilinear (
						images: graph.ExpandDims (
							input: graph.Cast (
								,
							dim: graph.Const (0, "make_batch")),
						size: graph.Const (new int [] { W, H }, "size")),
					y: graph.Const (Mean, "mean")),
				y: graph.Const (Scale, "scale"));
		}*/

		//
		// Downloads the inception graph and labels
		//
		static void ModelFiles (string dir)
		{
			string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz";

			modelFile = Path.Combine ("InceptionV3", "quantized_graph_v3.pb");
			labelsFile = Path.Combine ("InceptionV3", "imagenet_slim_labels.txt");
			//var zipfile = Path.Combine (dir, "inception_v3_2016_08_28_frozen.pb.tar.gz");

			/*if (File.Exists (modelFile) && File.Exists (labelsFile)) {
				File.Delete (modelFile);
				File.Delete (labelsFile);
			}*/

			//Directory.CreateDirectory (dir);
			//var wc = new WebClient ();
			//wc.DownloadFile (url, zipfile);
			//ZipFile.ExtractToDirectory (zipfile, dir);
			//File.Delete (zipfile);

		}

		public override void DidReceiveMemoryWarning ()
		{
			base.DidReceiveMemoryWarning ();
			// Release any cached data, images, etc that aren't in use.
		}
	}
}
