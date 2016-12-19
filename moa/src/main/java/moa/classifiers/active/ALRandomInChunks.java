package moa.classifiers.active;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;
import moa.options.ClassOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

public class ALRandomInChunks extends AbstractClassifier implements ALClassifier {

	private static final long serialVersionUID = 1L;
	
    @Override
    public String getPurposeString() {
        return "Random Active learning classifier for evolving data streams  with chunks";
    }
    
    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "drift.SingleClassifierDrift");
    
    public FloatOption budgetOption = new FloatOption("budget",
            'b', "Budget to use.",
            0.1, 0.0, 1.0);
    
    public IntOption chunkSizeOption = new IntOption("chunkSizeOption",
    		'c', "Size of chunk",
    		100, 1, 200);
    
    public Classifier classifier;
    
    public int lastLabelAcq = 0;
    
    public ArrayList<Instance> chunk = new ArrayList<Instance>(chunkSizeOption.getValue());
    


	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public int getLastLabelAcqReport() {
		int ret = lastLabelAcq;
		lastLabelAcq = 0;
		return ret;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		return this.classifier.getVotesForInstance(inst);
	}

	@Override
	public void resetLearningImpl() {
		this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.classifier.resetLearning();		
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		
		chunk.add(inst);
		if (chunk.size() < chunkSizeOption.getValue()) 	
			return;
			
		for (Instance instance : chunk) {
			if (this.budgetOption.getValue() > this.classifierRandom.nextDouble()) {
				this.classifier.trainOnInstance(instance);
				this.lastLabelAcq++;
			}
		}
		
		chunk.clear();
			
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		List<Measurement> measurementList = new LinkedList<Measurement>();
        return measurementList.toArray(new Measurement[measurementList.size()]);
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		((AbstractClassifier) this.classifier).getModelDescription(out, indent);		
	}
	
}
