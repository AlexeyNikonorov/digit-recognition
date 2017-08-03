#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>

using namespace std;

int read_data(const char *filename, vector< vector<double> >& data, vector<int>& target) {
	ifstream ifs(filename);
	while ( !ifs.eof() ) {
		int t;
		vector<double> d(64);
		ifs >> t;
		for (int i = 0; i < 64; i++) ifs >> d[i];
		data.push_back(d);
		target.push_back(t);
	}
	ifs.close();
	return data.size();
}

class LRC {
	enum { NFEATURES = 64 };
	double m_lr;
	double m_w[NFEATURES];
	double m_w0;
	vector< vector<double> > m_data;
	vector<int> m_target;
public:
	LRC() : m_lr(0.01) {}
	
	int predict(vector<double>& x) {
		return P(x, 1) > 0.5 ? 1 : -1;
	}
	
	void add_data(vector<double>& xt, int yt) {
		m_data.push_back(xt);
		m_target.push_back(yt);
	}
	
	void train() {
		int data_size = m_data.size();
		randomf(m_w, NFEATURES);
		randomf(&m_w0, 1);
		for (int i = 0; i < 10; i++)
			for (int t = 0; t < data_size; t++)
				update(m_data[t], m_target[t]);
	}
	
	void update(vector<double>& xt, int yt) {
		double p = 1.0 - P(xt, yt);
		double dw0 = m_lr * p * yt;
		m_w0 += dw0;
		for (int i = 0; i < NFEATURES; i++) m_w[i] += dw0 * xt[i];
	}
	
	double P(vector<double>& x, int y) {
		double inner = 0.0;
		for (int i = 0; i < NFEATURES; i++) inner += m_w[i] * x[i];
		return g(y * (inner + m_w0));
	}
	
	static void randomf(double *x, int n) {
		for (int i = 0; i < n; i++)
			x[i] = ((double) rand()) / RAND_MAX;
	}
	
	static double g(double x) {
		return 1.0 / (1.0 + exp(-x));
	}
};

class MultiClassOVR {
	enum { NCLASSES = 10 };
	LRC *m_clf[NCLASSES];
public:
	MultiClassOVR() {
		for (int i = 0; i < NCLASSES; i++) m_clf[i] = new LRC;
	}
	
	~MultiClassOVR() {
		for (int i = 0; i < NCLASSES; i++) delete m_clf[i];
	}
	
	int predict(vector<double>& x) {
		double max_prob = 0.0;
		int label = -1;
		for (int i = 0; i < NCLASSES; i++) {
			double prob = m_clf[i]->P(x, 1);
			if (prob > max_prob) {
				max_prob = prob;
				label = i;
			}
		}
		return label;
	}
	
	void train(vector< vector<double> >& data, vector<int>& target, int data_size) {
		for (int i = 0; i < data_size; i++) {
			int y = target[i];
			auto& x = data[i];
			for (int j = 0; j < NCLASSES; j++)
				if (j == y) m_clf[j]->add_data(x, 1);
				else m_clf[j]->add_data(x, -1);
		}
		for (int i = 0; i < NCLASSES; i++) m_clf[i]->train();
	}
};

class MultiClassOVO {
	enum { NCLS = 10 };
	vector< vector<LRC*> > m_clsf;
public:
	MultiClassOVO() : m_clsf(NCLS) {
		for (int i = 0; i < NCLS-1; i++)
			for (int j = i+1; j < NCLS; j++)
				m_clsf[i].push_back(new LRC);
	}

	~MultiClassOVO() {
		for (int i = 0; i < NCLS-1; i++)
			for (int j = i+1; j < NCLS; j++)
				delete m_clsf[i][j-i-1];
	}

	int predict(vector<double>& x) {
		vector<double> votes(NCLS, 0.0);
		for (int i = 0; i < NCLS; i++) {
			auto& clsf_list = m_clsf[i];
			int size = clsf_list.size();
			for (int j = 0; j < size; j++) {
				double pi = clsf_list[j]->P(x, -1);
				double pj = 1.0 - pi;
				votes[i] += pi;
				votes[i+1+j] += pj;
			}
		}
		double max_prob = 0.0;
		int label = -1;
		for (int i = 0; i < NCLS; i++) {
			if (votes[i] > max_prob) {
				max_prob = votes[i];
				label = i;
			}
		}
		return label;
	}

	void train(vector< vector<double> >& data, vector<int>& target, int data_size) {
		for (int i = 0; i < data_size; i++) {
			int y = target[i];
			auto& x = data[i];
			for (int j = 0; j < y; j++) m_clsf[j][y-j-1]->add_data(x, 1);
			for (int j = y+1; j < NCLS; j++) m_clsf[y][j-y-1]->add_data(x, -1);
		}
		for (auto& clsf_list : m_clsf)
			for (auto clsf : clsf_list)
				clsf->train();
	}
};

void show(vector<double>& x) { 
	for (int i = 0; i < 64; i++) {
		if (i % 8 == 0 && i != 0) cout << '\n';
		if (x[i] == 0) cout << "  .";
		else if (x[i] < 10) cout << "  " << (int) x[i];
		else cout << " " << (int) x[i];
	}
	cout << "\n\n";
}

int main(int argc, char *argv[]) {
	vector< vector<double> > data;
	vector<int> target;
	int n = read_data("data.txt", data, target);
	int nt = n / 2;
	
	srand(time(NULL));
	
	MultiClassOVO c;
	c.train(data, target, nt);
	
	int hits = 0, losses = 0;
	for (int i = nt; i < n; i++) {
		int ypred = c.predict(data[i]);
		int y = target[i];
		if (ypred == y) hits++;
		else losses++;
	}
	double acc = (double)(hits) / (hits + losses);
	cout << "accuracy on test data: " << acc << endl;
	
	if (argc > 1) {
		for (;;) {
			int i = nt + rand() % (n - nt);
			int ypred = c.predict(data[i]);
			int y = target[i];
			cout << "predicted: " << ypred << ", correct: " << y << endl;
			show(data[i]);
			cout << "Press \"Enter\" for next sample"; cin.get();
		}
	}
	return 0;
}