/* This file contains code from https://github.com/mimno/anchor,
 * which is distributed under MIT License, Copyright David Mimno */
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.List;
import java.util.Scanner;
import java.util.Comparator;
import java.util.Collections;

public class GibbsInference {
	static int numTopics; // Number of topics, determined from matrix

	static String doc_directory = 
			"synth";
	static String doc_truetopic_directory = 
			"synth_topics";

	static int r = 3; // sparsity parameter; alpha = r/numTopics
	static double alpha;
	static double alphaSum = r; // because alpha = r/numTopics
	static List<double[]> typeTopicWeights = new ArrayList<double[]>(); 
	static double[] topicSums;

    static final int burnin_iters = 200;
    static final int sampling_iters = 1000;

	public static double[] estimateTopics(String outputName,
			List<Integer> tokenSequence,
			int burnIn, 
			int samples, String dir_name, String file_name) throws IOException {

		int length = tokenSequence.size();
		int[] topicSequence = new int[length];			

		// initialize
		sampleTopicsForOneDoc (tokenSequence, topicSequence, true);

		// sample to burn-in
		for (int iteration = 0; iteration < burnIn; iteration++) {
			sampleTopicsForOneDoc (tokenSequence, topicSequence, false);
		}

		// Now start saving values
		int[] topicSums = new int[numTopics];
		for (int iteration = 0; iteration < samples; iteration++) {
			int[] topicCounts = sampleTopicsForOneDoc (tokenSequence, topicSequence, false);
			for (int topic = 0; topic < numTopics; topic++) {
				topicSums[topic] += topicCounts[topic];
			}
		}

		double normalizer = 1.0 / (length * samples + alphaSum);
		double[] result = new double[numTopics];
		for (int topic = 0; topic < numTopics; topic++) {
			result[topic] = (alpha + topicSums[topic]) * normalizer;
		}

        if (outputName != null) { 
		    PrintWriter out = new PrintWriter(outputName);
		    Formatter line = new Formatter();
		    for (int topic = 0; topic < numTopics; topic++) {
			    line.format("\t%f", (alpha + topicSums[topic]) * normalizer);
		    }
		    out.println(line);
		    out.close();
        }
        return result;
	}
	private static int compute_support_error(double [] truth,
			double[] result) throws FileNotFoundException {
		double l1_err = 0;
		double lmax_err = 0;
        List<Integer> gibbs_support = new ArrayList<Integer>(topN(r, result));
        List<Integer> true_support = new ArrayList<Integer>(topN(r, truth));
        List<Integer> intersection = new ArrayList<Integer>(gibbs_support);
        intersection.retainAll(true_support);
        int support_err = true_support.size() - intersection.size();
   		return support_err;
	}

	private static void compute_recovery_error(String dir_name,
			String file_name, double[] result, double[] sum_error) throws FileNotFoundException {
		String name = doc_truetopic_directory + "/" + dir_name + "/" + file_name;
		//System.err.println(name);
		Scanner s = new Scanner(new File(name));
		double[] truth = new double[result.length];
		int i = 0;
		while (s.hasNextDouble()) {
			truth[i++] = s.nextDouble();
		}
		double l1_err = 0;
		double lmax_err = 0;
		for (i = 0; i < truth.length; ++i) {
			double i_err = Math.abs(result[i] - truth[i]);
			l1_err += i_err;
			lmax_err = Math.max(lmax_err, i_err);
		}
		sum_error[0] += l1_err;
		sum_error[1] += lmax_err;
		s.close();
	}

	static int[] sampleTopicsForOneDoc (List<Integer> tokenSequence,
			int[] topicSequence, boolean initializing) {
		double[] currentTypeTopicWeights;
		int type, oldTopic, newTopic;
		int docLength = tokenSequence.size();

		int[] localTopicCounts = new int[numTopics];

		if (! initializing) {
			//		populate topic counts
			for (int position = 0; position < docLength; position++) {
				localTopicCounts[topicSequence[position]]++;
			}
		}

		double score, sum;
		double[] topicTermScores = new double[numTopics];

		//	Iterate over the positions (words) in the document 
		for (int position = 0; position < docLength; position++) {
			type = tokenSequence.get(position);
            //System.out.println(type);
			oldTopic = topicSequence[position];

			// Grab the relevant row from our two-dimensional array
			currentTypeTopicWeights = typeTopicWeights.get(type);

			if (! initializing) {
				//	Remove this token from all counts. 
				localTopicCounts[oldTopic]--;
			}

			// Now calculate and add up the scores for each topic for this word
			sum = 0.0;

			// Here's where the math happens! Note that overall performance is 
			//  dominated by what you do in this loop.
			for (int topic = 0; topic < numTopics; topic++) {
				score =
						(alpha + localTopicCounts[topic]) *
						(currentTypeTopicWeights[topic] / topicSums[topic]);
				sum += score;
				topicTermScores[topic] = score;
			}

			// Choose a random point between 0 and the sum of all topic scores
			double sample = Math.random() * sum;

			// Figure out which topic contains that point
			newTopic = -1;
			while (sample > 0.0) {
				newTopic++;
				sample -= topicTermScores[newTopic];
			}

			// Make sure we actually sampled a topic
			if (newTopic == -1) {
				throw new IllegalStateException ("DocTopicsSampler: New topic not sampled.");
			}

			// Put that new topic into the counts
			topicSequence[position] = newTopic;
			localTopicCounts[newTopic]++;
		}

		return localTopicCounts;
	}

    public static List<Integer> read_integers(File file) throws IOException {
   		Scanner ds = new Scanner(file);
   		List<Integer> document = new ArrayList<Integer>();
   		while (ds.hasNextInt()) {
   			document.add(ds.nextInt());
   		}
   		ds.close();	
        return document;
    }

    public static List<Integer> topN(int n, final double[] list) {
        List<Integer> indices = new ArrayList<Integer>(list.length);
        for (int i = 0; i < list.length; ++i) indices.add(i);
        // Want to sort indices by list score
        Collections.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer i_, Integer j_) {
                int i = i_;
                int j = j_;
                return Double.compare(list[i], list[j]);
            }
        });
        List<Integer> result = new ArrayList<Integer>();
        for (int i = 0; i < n; ++i) {
            result.add(indices.get(indices.size() - 1 - i));
        }
        return result;
    }

	public static final void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Need arguments: [infer | estimate-support] [topic-word-matrix-file].");
            System.exit(1);
        }

	    String matrix_file = args[1];
        Scanner word_counter = new Scanner(new File(matrix_file));
        String first_line = word_counter.nextLine();
        numTopics = first_line.split("\\s+").length;
        alpha = r/(double)numTopics;
        topicSums = new double[numTopics];
        word_counter.close();

		Scanner ms = new Scanner(new File(matrix_file));
		while (ms.hasNextDouble()) {
			typeTopicWeights.add(new double[numTopics]);
			for (int i = 0; i < numTopics; ++i) {
				typeTopicWeights.get(typeTopicWeights.size() - 1)[i] = ms.nextDouble();
				topicSums[i] += typeTopicWeights.get(typeTopicWeights.size() - 1)[i];
			}
		}
		System.out.println(typeTopicWeights.size());
		ms.close();

        if (args[0].equals("infer")) {
            /* Experiment, compare results with synthetic truth */
		    File folder = new File(doc_directory);
		    File[] subfolders = folder.listFiles();
		    for (File subfolder : subfolders) {
                double[] sum_error = new double[2];
		    	// subfolder corresponds to #words/doc
		    	String subfolder_name = subfolder.getName();
		    	File[] docs = subfolder.listFiles();
		    	for (File doc_file : docs) {
		    		String file_name = doc_file.getName();
                    List<Integer> doc = read_integers(doc_file);
		    		double[] result = estimateTopics(null,
		    				read_integers(doc_file), 
                            burnin_iters, sampling_iters,
                            subfolder_name, file_name);
                    
		            compute_recovery_error(subfolder_name, file_name, result, sum_error);
		    	}
		    	System.out.println(subfolder_name + "\t" + sum_error[0]/docs.length + "\t" + sum_error[1]/docs.length);
		    }
        } else if (args[0].equals("estimate-support")) {
            /* Experiment, estimate support of real document */
            File folder = new File ("real_doc");
            File[] docs = folder.listFiles();
            int total_intersection_size = 0;
            int total_gibbs_size = 0;
            int total_inversion_size = 0;
            for (File doc_file : docs) {
                String file_name = doc_file.getName();
                System.out.println("FILE: " + file_name);
                List<Integer> doc = read_integers(doc_file);
                double[] result = estimateTopics(null, //"real_doc" + "/" + file_name + ".inferred",
                            doc, burnin_iters, sampling_iters, "", ""); 
               // Compare TopN
                List<Integer> gibbs_support = new ArrayList<Integer>(topN(3, result));
                List<Integer> inversion_support = read_integers(new File("real_doc_support_guess/" + file_name));
                List<Integer> intersection = new ArrayList<Integer>(gibbs_support);
                intersection.retainAll(inversion_support);
                System.out.println("GIBBS SUPPORT SIZE:" + gibbs_support.size());
                System.out.println("INVERSION SUPPORT SIZE:" + inversion_support.size());
                System.out.println("INTERSECTION SIZE:" + intersection.size());
                total_intersection_size += intersection.size();
                total_gibbs_size += gibbs_support.size();
                total_inversion_size += inversion_support.size();
            }
            System.out.println("PRECISION: " + total_intersection_size/(double)total_inversion_size);
            System.out.println("RECALL: " + total_intersection_size/(double)total_gibbs_size);
        }
	}
}
