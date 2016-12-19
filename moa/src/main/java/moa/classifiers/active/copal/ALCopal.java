/*
 *    ALCopal2016.java
 *    Copyright (C) 2016 Otto von Guericke University, Magdeburg, Germany
 *    @author Serhii Vasylchenko (serhii dot vasylchenko at gmail dot com)
 *    @author Mariya Sotnikova (maria dot n dot sotnikova at gmail dot com)
 *    @author Uma Raju (umamrita at gmail dot com)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package moa.classifiers.active.copal;

import java.util.*;
import java.util.logging.*;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.active.ALClassifier;
import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.clusterers.Clusterer;
import moa.core.AutoExpandVector;
import moa.core.Measurement;
import moa.options.ClassOption;

import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.FloatOption;

/**
 *  Clustering-Based Probabilistic Active Learning (COPAL)
 *  
 *  Algorithm consists of three steps: clustering mechanism, macro step to choose the most beneficial 
 *  cluster and micro step to choose the most beneficial instance within this cluster. In the clustering 
 *  step pool of all instances is divided into some initial clusters. The main task is to establish the 
 *  initial neighborhoods of instances and not to achieve a good partitioning of the data space. In the 
 *  same step we try to split the clusters depending on class distribution. The idea of macro step consists 
 *  of choosing the most important cluster. To achieve that we use OPAL-gain formula from probabilistic 
 *  active learning [3]. In the micro step the most beneficial instance from the most beneficial cluster 
 *  is chosen. Then this new instance gets the label.
 *  
 *  Three variants of COPAL are implemented: basic, self-trained variant and ensemble classifier variant. 
 * 
 *  @author Serhii Vasylchenko (serhii dot vasylchenko at gmail dot com)
 *  @author Mariya Sotnikova (maria dot n dot sotnikova at gmail dot com)
 *  @author Uma Raju (umamrita at gmail dot com)
 *
 */
public class ALCopal extends AbstractClassifier implements ALClassifier {

	private static final long serialVersionUID = 1L;
	
	public MultiChoiceOption copalVariantOption = new MultiChoiceOption(
            "CopalVariant", 'v', "Copal Variant to use.", new String[]{
                "Basic", "SSL", "Ensemble classifier"}, new String[]{
                "Basic variant", 
                "Extended with semi-supervised learning",
                "Extended with ensemble classifier"}, 0);
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "bayes.NaiveBayes");
	public ClassOption mainClustererOption = new ClassOption("mainClusterer", 'c',
			"Clustering algorithm for pre-clustering", Clusterer.class, "clustream.WithKmeans");
	public ClassOption splitClustererOption = new ClassOption("splitClusterer", 'p',
			"Clustering algorithm for splitting", Clusterer.class, "clustream.WithKmeans -k 2");
	public FloatOption budgetOption = new FloatOption("budget",'b', "Budget to use.",
            0.1, 0.0, 1.0);
	public IntOption chunkSizeOption = new IntOption("chunkSize", 's',"size of chunk",100,1,200);
	public IntOption windowSizeOption = new IntOption("windowSize", 'w', "size of window" , 
			1000, 100, 2000);
	public FloatOption tauOption = new FloatOption("tau",'t', "relative cost of each false positive classification",
            0.5, 0.0, 1.0);
	
	public Classifier classifier;
	public Clusterer mainClusterer; // Used for initial clustering
	public Clusterer splitClusterer; // Used for splitting
	private Clustering currentClustering;
    
    public int lastLabelAcq = 0;    
    private long instNum = 0;   
    private long usedLabels = 0;
    private int splitDepth = 0;
	private int classNumber = 0;
    private static Logger log = Logger.getLogger(ALCopal.class.getName());
	private final int debug = 0; // 0 - no output; 1 - general output; 2 - detailed output; 3 - very detailed output
    
	public ArrayList<Instance> chunk = new ArrayList<Instance>();   
	public LinkedList<Instance> cache = new LinkedList<Instance>();
	public Map<Instance, Cluster> clusteringModel = new HashMap<Instance, Cluster>();
	public Map<Instance, Integer> labeledInstances = new HashMap<Instance, Integer>();

    
    //--------------------CLUSTERING-------------------------

	/**
	 * Clustering mechanism for COPAL. Uses MOA clusterers. Saves clustering as this.currentClustering
	 * 
	 * Every time after getting a new chunk, we use MOA clusterer  to get the initial clustrering from the scratch. All instances from cache and chunk are used for training the clusterer. Configuration settings of this clusterer also allow to manipulate the result. 
	 * After initial clustering is created, algorithm is trying to adjust clustering model according to class distribution of labeled instances in the clusters. We split every cluster, which fulfill next conditions:
	 * 	•cluster is inhomogeneous; 
	 * 	•every new cluster is big enough;
	 * 	•error is decreased.
	 * Cluster is considered inhomogeneous, if it has labeled instances with different cluster. The idea is, that splitting of such cluster should increase its homogeneity. Therefore, we calculate the current error rate E1 and compare it to the error rate after splitting E2. If the error rate decreases by splitting (i.e. E1 > E2), we replace old cluster with its subclusters.
	 * Also, to avoid unnecessary overfitting of clustering model, we need to limit the minimum number of instances in new clusters after splitting. Currently limitation is set to at least 5 points.
	 * The same splitting operations are performed on every new clusters recursively. Therefore, we will luckily achieve the best clustering model with current data.
	 */
	public void createClustering(){
		// Clustering using MOA clusterer
		if (this.debug > 0) System.out.println("----------------CLUSTERING----------------");
		this.currentClustering = new Clustering();
		this.mainClusterer.resetLearning();
		for (Instance inst : this.cache) {
			this.mainClusterer.trainOnInstance(inst);
		}
		for (Instance inst : this.chunk) {
			this.mainClusterer.trainOnInstance(inst);
		}
		if (this.debug > 1) System.out.println("Training the clusterer on " + (this.cache.size() + this.chunk.size()) + " instances is complete.");
		this.currentClustering = this.mainClusterer.getClusteringResult();
		
		if (this.currentClustering == null) {
			log.warning("Number of instances to be clustered: " + this.instNum + ", clusterer returned null");
			return;
		}
		
		//this.updateClusteringModel();

		if (this.debug > 1) {
			System.out.println("Initial clustering: ");
			this.printClusteringInfo(this.currentClustering);
		}
		
		// Splitting. This part maybe can be optimized
		if (this.debug > 1) System.out.println("----------------Splitting----------------");
		
		this.splitDepth = 0;
		this.trySplit(this.currentClustering.getClustering(), this.currentClustering);
		
		if (this.debug > 0) System.out.println("Final clustering:");
		this.printClusteringInfo(this.currentClustering);
				
	}
	
	/**
	 * Tries split every cluster. Conditions to split: 
	 * 		1) inhomogeneous (error is bigger than 0); 
	 * 		2) every new cluster is big enough (5 points);
	 * 		3) error is decreased.
	 * 
	 * @param clusters Vector of clusters to try split
	 * @param parentClustering Clustering, which clusters belong to
	 */
	public void trySplit(AutoExpandVector<Cluster> clusters, Clustering parentClustering) {
		AutoExpandVector<Cluster> newClusters; // new clusters after each splitting
		AutoExpandVector<Cluster> allNewClusters = new AutoExpandVector<Cluster>(); // new clusters after all splittings on this level
		
		double oldError, newError;

		if (this.debug > 1) System.out.println("SPLITTING : depth " + this.splitDepth);
		
		for (int i = 0; i < clusters.size(); i++) { // for every cluster we want to try split
			
			Cluster oldCluster = clusters.get(i);			
			oldError = this.getClusterError(oldCluster, parentClustering);
			if (this.debug > 1) {
				System.out.println("Cluster " + i + ":");
				System.out.println("--Class distribution: " + Arrays.toString(this.getClassDistribution(oldCluster, parentClustering)));
				System.out.print("--Error of this cluster: " + oldError);
			}

			if (oldError > 0) { // First condition - inhomogeneity
				if (this.debug > 1) System.out.println(". Inhomogeneous, try to split");
				
				newClusters = this.split(oldCluster, parentClustering);
				if (this.debug > 1) System.out.println("--Cluster is splitted into " + newClusters.size() + " new clusters:");
				if (newClusters.size() < 2) { 
					// something wrong with splitting clusterer, skip this cluster
					continue;
				}
				
				// We need to create another clustering object, where will be new clusters instead of old one.
				// Otherwise we could not get the points, which will be assign to new clusters 
				// and therefore would not be able to calculate the new error.
				Clustering newClustering = new Clustering();
				newClustering = (Clustering) parentClustering.copy();
				newClustering.remove(i);											
				for (int j = 0; j < newClusters.size(); j++) {
					newClustering.add(newClusters.get(j));					
				}
				
				// Second condition - new clusters should not be too small
				boolean tooSmall = false;
				for (int j = 0; j < newClusters.size(); j++) {
					if (this.debug > 1) System.out.println("----Cluster " + j + " (" + this.getClusterPoints(newClusters.get(j), newClustering).size() + " points) : " + Arrays.toString(newClusters.get(j).getCenter()));
					int num = this.getClusterPoints(newClusters.get(j), newClustering).size();
					if (num < 5) {
						if (this.debug > 1) System.out.println("----New cluster " + j + " is too small (" + num + "), cancel splitting");
						tooSmall = true;
						break;
					}
				}
				if (tooSmall)
					continue; // If any of new clusters is too small, skip it

				if (this.debug > 1) {
					System.out.println("--Error of old cluster: " + oldError + ", error of new clusters:");
					System.out.print("--");
				}
				newError = 0;
				int num = 0; // Number of new clusters with labeled instances inside for the correct calculation of error
				double error; // This variable is only for debugging sake
				for (int j = 0; j < newClusters.size(); j++) {
					if (this.debug > 1) System.out.print("Cluster " + j + ": ");
					if (!this.hasLabeledInstances(newClusters.get(j), newClustering)) {
						if (this.debug > 1) System.out.print("(no labeled instances), ");
						break;
					}
					error = this.getClusterError(newClusters.get(j), newClustering);
					newError += error;
					if (this.debug > 1) System.out.print(error + ", ");
					num++;
				}
				if (num == 0) {
					if (this.debug > 1) System.out.println("--New clusters have no labeled instances, something should be wrong");
					continue;
				}
				newError = newError / num;
				if (this.debug > 1) System.out.println("mean: " + newError);
				
				if (oldError > newError) { // Third condition - error is decreased
					this.currentClustering = (Clustering) newClustering.copy();

					if (this.debug > 1) System.out.println("--Error is decreased, old cluster is splitted. New clustering replaces the old one:");
					this.printClusteringInfo(this.currentClustering);
					
					for (int j = 0; j < newClusters.size(); j++) {
						allNewClusters.add(newClusters.get(j));
					}
				}
			}
			else
				if (this.debug > 1) System.out.println(". Homogeneous, go to the next one");
		}
		
		if (allNewClusters.size() > 0) { // Try splitting on all new clusters
			this.splitDepth++;
			this.trySplit(allNewClusters, this.currentClustering);
		}
		
	}

	/**
	 * Split cluster. Trains the splitClusterer on all points from this cluster.
	 *
	 * @param cluster Cluster to split
	 * @param parentClustering Clustering, which cluster belongs to
	 * @return AutoExpandVector<Cluster> of new clusters
     */
	private AutoExpandVector<Cluster> split(Cluster cluster, Clustering parentClustering) {
		
		this.splitClusterer.resetLearning();
		for (Instance inst : getClusterPoints(cluster, parentClustering)) {
			this.splitClusterer.trainOnInstance(inst);
		}
		
		return this.splitClusterer.getClusteringResult().getClustering();
	}
	
	/**
	 * Calculates classification error in the cluster
	 * 
	 * @param cluster
	 * @param parentClustering Clustering, which cluster belong to
	 * @return double value of an error
	 */
	public double getClusterError(Cluster cluster, Clustering parentClustering) {
		
		int num = 0;
		int[] classDistribution = new int[this.classNumber];
		
		for (Instance inst : getClusterPoints(cluster, parentClustering)) {
			if (this.labeledInstances.containsKey(inst)) {
				classDistribution[this.labeledInstances.get(inst)]++;
				num++;
			}
		}
		
		if (num > 0) {
			double mainClassNum = 0;
			
			for (int i = 0; i < classDistribution.length; i++) {
				if (classDistribution[i] > mainClassNum) {
					mainClassNum = classDistribution[i];
				}
			}
			return (1 - mainClassNum / num);
		}
		return 0;
	}

	/**
	 * Calculates classification error in the cluster, using currentClustering as parent clustering
	 * 
	 * @param cluster
	 * @return double value of an error
	 */
	public double getClusterError(Cluster cluster) {
		return this.getClusterError(cluster, this.currentClustering);
	}
	
	/**
	 * Check if there are labeled instances in the cluster
	 * 
	 * @param cluster
	 * @param parentClustering Clustering, which cluster belong to
	 * @return true, if there are labeled instances in the cluster
	 */
	public boolean hasLabeledInstances(Cluster cluster, Clustering parentClustering) {
		for (Instance inst : getClusterPoints(cluster,parentClustering)) {
			if (this.labeledInstances.containsKey(inst)) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Calculates distribution among all classes in the cluster
	 *
	 * @param cluster Cluster
	 * @param parentClustering Clustering, which cluster belong to
     * @return Integer array, index - class index, value - number of instances with such class
     */
	public int[] getClassDistribution(Cluster cluster, Clustering parentClustering) {
		int[] classDistribution = new int[this.classNumber];

		for (Instance inst : getClusterPoints(cluster, parentClustering)) {
			if (this.labeledInstances.containsKey(inst)) {
				classDistribution[this.labeledInstances.get(inst)]++;
			}
		}
		return classDistribution;
	}
	
	/**
	 * Get all points from a cluster
	 * 
	 * @param cluster Cluster
	 * @param parentClustering Clustering, which cluster belong to
	 * @return ArrayList of all points, associated with this cluster
	 */
	public ArrayList<Instance> getClusterPoints(Cluster cluster, Clustering parentClustering) {
		// Recalculates every time with current implementation, 
		// may be better to save it once and only access the data
		ArrayList<Instance> instances = new ArrayList<Instance>();	

		double maxProb, prob;
		Cluster bestCluster = null;
		for (Instance inst : this.cache) {
			maxProb = 0;
			for (Cluster cluster1 : parentClustering.getClustering()) {
				prob = cluster1.getInclusionProbability(inst);
				if (prob > maxProb) {
					bestCluster = cluster1;
					maxProb = prob;
					}
			}
			if (bestCluster == cluster) {
				instances.add(inst);
			}
		}
		for (Instance inst : this.chunk) {
			maxProb = 0;
			for (Cluster cluster1 : parentClustering.getClustering()) {
				prob = cluster1.getInclusionProbability(inst);
				if (prob > maxProb) {
					bestCluster = cluster1;
					maxProb = prob;
					}
			}
			if (bestCluster == cluster) {
				instances.add(inst);
			}
		}
		
		return instances;
	}
	
	/**
	 * Get all points from a cluster, using this.currentClustering as parent clustering
	 * 
	 * @param cluster Cluster
	 * @return ArrayList of all points, associated with this cluster
	 */
	public ArrayList<Instance> getClusterPoints(Cluster cluster) {
		return this.getClusterPoints(cluster, this.currentClustering);
	}
	
	@SuppressWarnings("unused")
	private void updateClusteringModel() { // Not used now, may be used in the future. Current implementation may work incorrectly
		// Clustering model - associate every instance with some class
		this.clusteringModel = new HashMap<Instance, Cluster>();
		double maxProb, prob;
		for (Instance inst : this.cache) {
			maxProb = 0;
			for (Cluster cluster : this.currentClustering.getClustering()) {
				prob = cluster.getInclusionProbability(inst);
				if (prob >= maxProb) {
					this.clusteringModel.put(inst, cluster);
					maxProb = prob;
					}
			}
		}
		for (Instance inst : chunk) {
			maxProb = 0;
			for (Cluster cluster : this.currentClustering.getClustering()) {
				prob = cluster.getInclusionProbability(inst);
				if (prob >= maxProb) {
					this.clusteringModel.put(inst, cluster);
					maxProb = prob;
					}
			}
		}
	}
	
	
	/**
	 * Print information about clustering
	 * 
	 * @param clustering
	 * @param detailed Print detailed information (points from every cluster)
	 */
	private void printClusteringInfo(Clustering clustering) {
		if (this.debug > 0) {
			Cluster cluster;
			for (int j=0; j < clustering.size(); j++) {
				cluster = clustering.get(j);
				System.out.println("-Cluster " + j + " (" + this.getClusterPoints(cluster, clustering).size() + " points) : " + Arrays.toString(cluster.getCenter()));
				System.out.println("--Class distrubution: " + Arrays.toString(this.getClassDistribution(cluster, clustering)));
				if (this.debug > 2) {
					System.out.println("--Cluster points (distances): ");
					for (Instance inst : getClusterPoints(cluster, clustering)) {
						System.out.println("----" + this.euclideanDistance(this.inputAttributes(inst), cluster.getCenter()));
						if (this.labeledInstances.containsKey(inst)){
							System.out.println("-----Instance is labeled with " + this.labeledInstances.get(inst));
						}				
					}
				}
			}
		}
	}
	
	//--------------------------------MACRO STEP--------------------------------
	
	 /** Determine the most beneficial cluster in current clustering
	 * 
	 * @return cluster
	 */
	private Cluster macroStep() {
		Cluster beneficialCluster = null;
		double weightOfCluster = 0;
		Boolean firstCluster = true;
		int numberInstancesAllClusters = 0;

		// Calculate number of instances in all clusters
		for (Cluster cluster : this.currentClustering.getClustering()) {
			numberInstancesAllClusters += this.getClusterPoints(cluster).size();
		}

		for (Cluster cluster : this.currentClustering.getClustering()) {
			// Initializing of variable beneficialCluster
			if (firstCluster) {
				beneficialCluster = cluster;
				firstCluster = false;
			}

			int numberInstancesCurrentCluster = this.getClusterPoints(cluster).size();

			// Calculation of gOPAL and weight of cluster. According to paper of Krempl and Ha
			double gOPAL = getGOPAL(cluster);
			double temp = getWeightGain(gOPAL, numberInstancesAllClusters, numberInstancesCurrentCluster);
			
			if (this.debug > 1) { System.out.println("Weight of the cluster " + temp); }

			// Saving the most beneficial cluster till this moment and its weight
			if (temp > weightOfCluster) {
				weightOfCluster = temp;
				beneficialCluster = cluster;
			}
		}
		return beneficialCluster;
	}

	/** Expected average misclassification loss reduction in the cluster
	 * 
	 * @param cluster
	 * @return result of gOPAL formula
	 */
	private double getGOPAL(Cluster cluster) {
		int numberLabeledInstances = 0;
		int numberPosInst = 0;

		// Calculate number of labeled instances
		for (Instance inst : getClusterPoints(cluster)) {
			if (this.labeledInstances.containsKey(inst)) {
				numberLabeledInstances++;
				if (this.labeledInstances.get(inst) == 1)
					numberPosInst++;
			}
		}

		double sharePosInst = 0;
		if (numberPosInst > 0) {
			sharePosInst = numberPosInst / numberLabeledInstances;
		}

		// Get values from GUI
		double m = budgetOption.getValue();
		double tau = tauOption.getValue();

		double np = numberLabeledInstances * sharePosInst;
		double factorilaCoefficient = factorial(numberLabeledInstances)
				/ (factorial(np) * factorial(numberLabeledInstances - np));

		double imlSum = 0;
		for (int i = 0; i <= m; i++) {
			imlSum = imlSum + iml(numberLabeledInstances, sharePosInst, tau, m, i);
		}

		double result = ((numberLabeledInstances + 1) / m) * factorilaCoefficient
				        * iml(numberLabeledInstances, sharePosInst, tau, 0, 0);
		return result;
	}

	/**Iml
	 * 
	 * @param numberLabeledInstances
	 * @param sharePosInst - share of positive instances among labeled one in the current cluster
	 * @param tau - relative cost of each false positive classification
	 * @param m - budget to use
	 * @param k incremental variable from gOPAL formula
	 * @return result of Iml formula
	 */
	private double iml(int numberLabeledInstances, double sharePosInst, double tau, double m, int k) {
		double result = 0;
		double factorilaCoefficient = factorial(m) / (factorial(k) * factorial(m - k));
		if ((m + numberLabeledInstances) == 0) {
			double g1 = gamma(2 - k + m + numberLabeledInstances - numberLabeledInstances * sharePosInst);
			double g2 = gamma(1 + k + numberLabeledInstances * sharePosInst);
			double g3 = gamma(3 + m + numberLabeledInstances);
			double gammaIntermediateCalculation = g1 * g2 / g3;
			result = factorilaCoefficient * (tau) * gammaIntermediateCalculation;
			return result;
		} else {
			double condition = (numberLabeledInstances * sharePosInst + k) / (m + numberLabeledInstances);
			if (condition < tau) {
				double g1 = gamma(1 - k + m + numberLabeledInstances - numberLabeledInstances * sharePosInst);
				double g2 = gamma(2 + k + numberLabeledInstances * sharePosInst);
				double g3 = gamma(3 + m + numberLabeledInstances);
				double gammaIntermediateCalculation = g1 * g2 / g3;
				result = factorilaCoefficient * (1 - tau) * gammaIntermediateCalculation;
				return result;
			} else if (condition == tau) {
				double g1 = gamma(1 - k + m + numberLabeledInstances - numberLabeledInstances * sharePosInst);
				double g2 = gamma(1 + k + numberLabeledInstances * sharePosInst);
				double g3 = gamma(2 + m + numberLabeledInstances);
				double gammaIntermediateCalculation = g1 * g2 / g3;
				result = factorilaCoefficient * (tau - tau * tau) * gammaIntermediateCalculation;
				return result;
			} else {// (condition > tau)
				double g1 = gamma(2 - k + m + numberLabeledInstances - numberLabeledInstances * sharePosInst);
				double g2 = gamma(1 + k + numberLabeledInstances * sharePosInst);
				double g3 = gamma(3 + m + numberLabeledInstances);
				double gammaIntermediateCalculation = g1 * g2 / g3;
				result = factorilaCoefficient * (tau) * gammaIntermediateCalculation;
				return result;
			}
		}
	}

	/** Calculating of log gamma function
	 * 
	 * @param x to calculate log gamma function from
	 */
	private double logGamma(double x) {
		double tmp = (x - 0.5) * Math.log(x + 4.5) - (x + 4.5);
		double ser = 1.0 + 76.18009173 / (x + 0) - 86.50532033 / (x + 1) + 24.01409822 / (x + 2) - 1.231739516 / (x + 3)
				+ 0.00120858003 / (x + 4) - 0.00000536382 / (x + 5);
		return (tmp + Math.log(ser * Math.sqrt(2 * Math.PI)));
	}

	/** Calculating of gamma function
	 * 
	 * @param x to calculate gamma function from
	 */
	private double gamma(double x) {
		return Math.exp(logGamma(x));
	}

	/** Computing of weighted gain. Cluster-size-weighted probabilistic gain
	 * 
	 * @param gOPAL result of formula. Expected average misclassification loss reduction in the cluster
	 * @param numberInstancesAllClusters
	 * @param numberInstancesCurrentCluster
	 * @return
	 */
	double getWeightGain(double gOPAL, int numberInstancesAllClusters, int numberInstancesCurrentCluster) {
		double weightCluster = gOPAL * numberInstancesCurrentCluster / numberInstancesAllClusters;
		return weightCluster;
	}

	/** Calculate factorial of value
	 * 
	 * @param n to calculate factorial from
	 * @return factorial of value
	 */
	public long factorial(double n) {
		long ans = 1;
		long max = 8223372036854775806L;// 9223372036854775806L;
		for (int i = 1; i <= n; i++) {
			if (ans >= max) {
				return ans;
			}
			ans = ans * i;
		}
		return ans;
	}

	/** Choose randomly cluster from clustering
	 * 
	 * @return cluster
	 */
	@SuppressWarnings("unused")
	private Cluster macroStepRandom() {
		if (this.currentClustering.size() == 1)
			return this.currentClustering.get(0);
		if (this.currentClustering.size() > 1)
			return this.currentClustering.get(this.classifierRandom.nextInt(this.currentClustering.size() - 1));
		return null;
	}

	//--------------------------------MICRO STEP--------------------------------
	
	/**
	 * Determines most beneficial instance in the cluster. It's the one, 
	 * which is the furthest away from its nearest obtained labeled instance.
	 * 
	 * @param cluster Cluster
	 * @return Most beneficial label
	 */
	private Instance microStep(Cluster cluster) {
		//EuclideanDistance ed = new EuclideanDistance();
		ArrayList<Instance> UnlabelledInstanceArray = new ArrayList<Instance>();
		ArrayList<Instance> LabelledInstanceArray = new ArrayList<Instance>();
		int pos = 0;
		//hashmap<integer,string> dm = new hashmap<integer,string>();
	 
				
		// for loop separates labelled and unlabelled instances
		for(Instance inst : getClusterPoints(cluster))
		{
		
		  if(this.labeledInstances.containsKey(inst))
		   {
			  LabelledInstanceArray.add(inst);
			  
		   }
		   else
		    {
		     
		     UnlabelledInstanceArray.add(inst);
		     
		    }
		  }
		  
		 				
		if(LabelledInstanceArray.isEmpty())
		{
			if(UnlabelledInstanceArray.isEmpty())
			{
				return null;
			}
			  if(UnlabelledInstanceArray.size()==1)
			 {
				return UnlabelledInstanceArray.get(0);
			 }
			 else
			 {
				return UnlabelledInstanceArray.get(this.classifierRandom.nextInt(UnlabelledInstanceArray.size()-1));
			 }
		}
		
		if(UnlabelledInstanceArray.isEmpty())
		{
			return null;
		}
		  ArrayList<Double> minDistValue = new ArrayList<Double>();
		  Map<Integer, Double> values = new HashMap<Integer, Double>();
		 
		//calculates the euclidean distance between each unlabelled instance to all labelled instances
		for(int i=0;i<UnlabelledInstanceArray.size();i++)
		{
			   double[] instance1Array = this.inputAttributes(UnlabelledInstanceArray.get(i));
			for(int j=0;j<LabelledInstanceArray.size();j++)
			{
				 double[] instance2Array = this.inputAttributes(LabelledInstanceArray.get(j));
				 double dist = euclideanDistance(instance1Array, instance2Array);
					minDistValue.add(dist);
					 pos = i;
				 				
			}
			Collections.sort(minDistValue);
			double m = minDistValue.get(0);
			values.put(pos, m);
			
			
			
		}
		 Map<Integer, Double> sortedMap = sortByComparator(values);
		 int key =0;
		    for(Map.Entry<Integer, Double> entry : sortedMap.entrySet())
		    {
		        key=entry.getKey();
		    }
		    
			return  UnlabelledInstanceArray.get(key);
	}
	
	private static Map<Integer, Double> sortByComparator(Map<Integer, Double> values) {
		
		// Convert Map to List
				List<Map.Entry<Integer, Double>> list = 
					new LinkedList<Map.Entry<Integer, Double>>(values.entrySet());

				// Sort list with comparator, to compare the Map values
				Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
					public int compare(Map.Entry<Integer, Double> o1,
		                                           Map.Entry<Integer, Double> o2) {
						return (o1.getValue()).compareTo(o2.getValue());
					}
				});

				// Convert sorted map back to a Map
				Map<Integer, Double> sortedMap = new LinkedHashMap<Integer, Double>();
				for (Iterator<Map.Entry<Integer, Double>> it = list.iterator(); it.hasNext();) {
					Map.Entry<Integer, Double> entry = it.next();
					sortedMap.put(entry.getKey(), entry.getValue());
				}
				return sortedMap;
		
	}	
	
	/**
	 * Random selection of the most beneficial instance within the cluster.
	 * 
	 * @param cluster Cluster
	 * @return The most beneficial instance
	 */
	@SuppressWarnings("unused")
	private Instance microStepRandom(Cluster cluster) {
		if (this.getClusterPoints(cluster).size() == 1)
			return this.getClusterPoints(cluster).get(0);
		if (this.getClusterPoints(cluster).size() > 1)
			return this.getClusterPoints(cluster).get(this.classifierRandom.nextInt(this.getClusterPoints(cluster).size()));
		return null;
	}
	
	//----------------------------------------------------------------------------
	
	/**
	 * This function implements cache management for COPAL. It adds new chunk to the cache and deletes old instances.
	 * If old instance was labeled, this information is also deleted. 
	 */
	private void saveToCache() {
		int n = 0;
		for (int i = 0; i < this.chunk.size(); i++) {
			this.cache.add(this.chunk.get(i));
		}
		while (this.cache.size() > this.windowSizeOption.getValue()) {
			if (this.labeledInstances.containsKey(this.cache.getFirst())) {
				this.labeledInstances.remove(this.cache.getFirst());
				n++;
			}
			this.cache.removeFirst();
		}
		if (this.debug > 1) System.out.println("Labeled instances deleted: " + n);
	}
	
	/**
	 * Implements self-labeling mechanism. If prediction of some class is more than epsilon, 
	 * instance is labeled with this class and weight 0.5 and is used for training.
	 */
	private void selfLabeling() {
		if (this.debug > 0) System.out.println("----------------SEMI-SUPERVISED LEARNING----------------");  
		double e = 0.25; // strange value because of current formula to calculate prediction
		double[] prediction;
		for (Instance inst : this.chunk) {
			if (!this.labeledInstances.containsKey(inst)) {
				prediction = this.getVotesForInstanceClusteringBased(inst);
				if (this.debug > 1) System.out.println("Class prediction: " + Arrays.toString(prediction));
				for (int i = 0; i < prediction.length; i++) {
					if (prediction[i] >= e) {
						if (this.debug > 0) System.out.println("Instance is labeled with " + i + " (" + prediction[i] + ")");
						inst.setClassValue((double) i);
						inst.setWeight(0.5);
						this.labeledInstances.put(inst, i);
						this.classifier.trainOnInstance(inst);
					}
				}
			}			
		}
	}
	
	/**
	 * Calculates class prediction for instance based on current clustering model
	 * 
	 * @param inst Instance
	 * @return Array of double values with predictions for every class
	 */
	private double[] getVotesForInstanceClusteringBased(Instance inst) {
		double[] prediction = new double[inst.numClasses()];
		Arrays.fill(prediction, 0);
		
		if (this.currentClustering == null || this.currentClustering.size() == 0) {
			log.warning("Current clustering is null or has no clusters, no prediction possible");
			return prediction;
		}			
		
		double share; // share of same class in he cluster
		double dist; // distance to the cluster
		int labelsNum, classNum; // to calculate share
		double dist_sum = 0; // sum of distances to all clusters
		double[] instPosition = this.inputAttributes(inst); // coordinates of the instance

		for (int i = 0; i < this.currentClustering.size(); i++) {
			dist_sum += this.euclideanDistance(instPosition, this.currentClustering.get(i).getCenter());
		}
		for (int i = 0; i < prediction.length; i++) {
			
			if (this.debug > 2) System.out.print("Prediction for class " + i + ": ");
			share = 0;
			dist = 0;
			for (int j = 0; j < this.currentClustering.size(); j++) {
				double[] clusterCenter = this.currentClustering.get(j).getCenter();
				dist = this.euclideanDistance(instPosition, clusterCenter);
				
				labelsNum = 0;
				classNum = 0;
				for (Instance clusterInst : getClusterPoints(this.currentClustering.get(j))) {
					if (this.labeledInstances.containsKey(clusterInst)) {
						labelsNum++;
						if (this.labeledInstances.get(clusterInst) == i)
							classNum++;				
					}
				}
				if (classNum > 0) {
					share = (double) classNum / labelsNum;
					}
				prediction[i] += share * (1 / (1 + dist)) / dist_sum;
				if (this.debug > 2) System.out.print(" + " + share + " * (1 / (1 + " + dist + ")) / " + dist_sum);
				}
			if (this.debug > 2) System.out.print("\n");
			}
		
		return prediction;
	}
	
	/**
	 * Input attributes of the instance
	 * 
	 * @param inst Instance
	 * @return Input attributes
	 */
	public double[] inputAttributes(Instance inst)
	{
		int i = 0;
		int num = inst.numInputAttributes();
		double[] array = new double[num];
		for(i=0; i<num;i++)
		{
			 array[i] = inst.value(i);
		}
		return array;
	}
	
	/**
	 * Calculates euclidean distance between two points
	 * 
	 * @param inst1coord First point
	 * @param inst2coord Second point
	 * @return euclidean distance
	 */
	public double euclideanDistance(double[] inst1coord , double[] inst2coord)
	{
		double sum = 0.0;
		for(int i= 0; i<inst1coord.length; i++)
		{
			sum = sum + Math.pow((inst1coord[i] - inst2coord[i]) , 2);
		}
		return Math.sqrt(sum);
		
	}
	
	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public int getLastLabelAcqReport() {
		int ret = this.lastLabelAcq;
		this.lastLabelAcq = 0;
		return ret;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		if (this.copalVariantOption.getChosenIndex() == 2) { // With ensemble classifier
			return this.getVotesForInstanceClusteringBased(inst);
		}
		return this.classifier.getVotesForInstance(inst);
	}

	@Override
	public void resetLearningImpl() {
		this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.classifier.resetLearning();
        this.mainClusterer = ((Clusterer) getPreparedClassOption(this.mainClustererOption)).copy();
        this.mainClusterer.resetLearning();
        this.splitClusterer = ((Clusterer) getPreparedClassOption(this.splitClustererOption)).copy();
        this.splitClusterer.resetLearning();
        this.chunk = new ArrayList<Instance>();
        this.cache = new LinkedList<Instance>();
        this.clusteringModel = new HashMap<Instance, Cluster>();
        this.instNum = 0;
        this.usedLabels = 0;
        this.splitDepth = 0;
    	this.classNumber = 0;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (this.classNumber == 0) this.classNumber = inst.numClasses();
		// Add new instances to the chunk until it's full
		this.chunk.add(inst);
		this.instNum++;
		//System.out.println(this.chunk.size() + " :: " + Arrays.toString(inst.toDoubleArray()));
		if (this.chunk.size() < this.chunkSizeOption.getValue()) 
			return; // quit if chunk is not full yet
		if (this.debug > 0) System.out.println("\n----------------START TRAINING----------------");
		if (this.debug > 0) System.out.println("Chunk size: " + this.chunk.size() + ", Cache size: " + this.cache.size());
		if (this.debug > 0) System.out.println("Instances processed: " + this.instNum);
		
		long availableLabels = (long) (this.budgetOption.getValue() * this.instNum);
		if (this.debug > 0) System.out.println("Labels available: " + availableLabels + ", labels used: " + this.usedLabels);
		if (availableLabels == 0)
			return; // quit if no budget available
		
		if (this.debug > 0) System.out.println("Current number of labeled instances in cache: " + this.labeledInstances.size());
		
		createClustering();
		if (this.currentClustering == null || this.currentClustering.size() == 0) {
			log.warning("Clustering is not existing or has no clusters in it, no training possible");
			return;
		}

		if (this.debug > 0) System.out.println("----------------ACTIVE LEARNING----------------");
		int failsNum = 0;
		Cluster chosenCluster; 
		Instance toBeLabeled;
		while (this.usedLabels < availableLabels) {
			if (failsNum > 20) {
				break;
			}
			chosenCluster = this.macroStep();
			if (chosenCluster == null) {
				log.warning("Macro step returned null, no training is possible");
				failsNum++;
				continue;
			}
			if (this.debug > 0) System.out.println("Chosen cluster: " + Arrays.toString(chosenCluster.getCenter()));
			toBeLabeled = this.microStep(chosenCluster);
			if (toBeLabeled == null) {
				log.warning("Micro step returned null, no training is possible");
				failsNum++;
				continue;
			}
			if (this.labeledInstances.containsKey(toBeLabeled)) {
				log.warning("Instance is already labeled, no training is possible");
				failsNum++;
				continue;
			}
			if (this.debug > 0) System.out.println("Chosen instance: " + Arrays.toString(toBeLabeled.toDoubleArray()));
			
			// Imitating labeling from an expert
			this.labeledInstances.put(toBeLabeled, (int) toBeLabeled.classValue());
			this.usedLabels++;
			
			if (this.copalVariantOption.getChosenIndex() != 2) { // Not ensemble classifier version
				this.classifier.trainOnInstance(toBeLabeled); // Train the base classifier
			}

		}
		if (this.copalVariantOption.getChosenIndex() == 1) { // With semi-supervised learning
			selfLabeling();
		}
		
		saveToCache();
		this.chunk = new ArrayList<Instance>();	
		if (this.debug > 0) System.out.println("----------------END OF CHUNK----------------");
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		List<Measurement> measurementList = new LinkedList<Measurement>();
		measurementList.add(new Measurement("number of labeled instances", this.labeledInstances.size()));
		
		Measurement[] modelMeasurements = ((AbstractClassifier) this.classifier).getModelMeasurements();
        if (modelMeasurements != null) {
            for (Measurement measurement : modelMeasurements) {
                measurementList.add(measurement);
            }
        }
		return measurementList.toArray(new Measurement[measurementList.size()]);
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		((AbstractClassifier) this.classifier).getModelDescription(out, indent);
	}

}