#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <float.h>
#include <algorithm>
#include "ANN.cpp"


using namespace Eigen;

MatrixXd getTril(MatrixXd& mat) {
	MatrixXd tril = mat.triangularView<Lower>();
	for (int i = 0; i < tril.rows(); i++)
		tril(i,i) = 0;
	return tril;
}

void appendVec2Mat(MatrixXd& mat, VectorXd vec) {
	mat.conservativeResize(mat.rows(), mat.cols()+1);
	mat.col(mat.cols()-1) = vec;
}

VectorXd StdVec2Eigen(std::vector <double> vec){
  VectorXd outvec(vec.size());
  for(int i=0;i<vec.size();i++)
    outvec(i)=vec[i];
  return outvec;
}

std::vector <double> EigenVec2Std(VectorXd vec){
  std::vector <double> outvec(vec.rows(),0.0);
  for(int i=0;i<vec.rows();i++)
    outvec[i]=vec(i);
  return outvec;
}


class ObjFunc{
public:
  	void init(ANN& input) {
  		nnet = input;
  		target = VectorXd::Constant(20,0);
  	}
  	void setTarget(VectorXd input) {
  		for (int i = 0; i < 10; i++) {
  			target(2*i) = input(i);
  			target(2*i + 1) = input(i);
  		}
  	}
 //  	VectorXd evalANN(VectorXd input){
 //  		std::vector<double> temperature = nnet.ComputeANN(EigenVec2Std(input));
	// 	VectorXd output(10);
	// 	for(int i=0;i<10;i++){
	// 		output(i)=	0.5*(0.15*temperature[3*i] + 
	// 					0.85*temperature[3*i+1] + 
	// 					0.85*temperature[3*i+2] + 
	// 					0.15*temperature[3*i+3]);
	// 	}
	// 	return output;
	// }
  	VectorXd evalANN(VectorXd input){
  		std::vector<double> temperature = nnet.ComputeANN(EigenVec2Std(input));
		VectorXd output(20);
		for(int i=0;i<10;i++){
			output(2 * i) = (temperature[3*i] + temperature[3*i + 1]) * 0.015 / 2 + 
			(1.5 * temperature[3*i + 1] + 0.5 * temperature[3*i + 2]) * 0.035 / 2;
			output(2 * i) /= 0.05;
			output(2 * i + 1) = (temperature[3*i + 2] + temperature[3*i + 3]) * 0.015 / 2 + 
			(0.5 * temperature[3*i + 1] + 1.5 * temperature[3*i + 2]) * 0.035 / 2;
			output(2*i + 1) /= 0.05;
		}
		return output;
	}
	void eval(double& f, VectorXd& g, VectorXd input){
		VectorXd output = evalANN(input);
		// std::cout << output << std::endl;
		int n = input.rows();
		f = (output - target).dot(output - target);
		VectorXd gradient = VectorXd::Constant(n,0.0);
		double delta = 0.01;
		// std::cout << "BP\n";
		for (int i = 0; i < n; i++) {
			VectorXd d_input = VectorXd::Constant(n,0.0);
			d_input(i) = delta;
			VectorXd temp_output1 = evalANN(input + d_input);
			VectorXd temp_output2 = evalANN(input - d_input);
			gradient(i) = (temp_output1 - target).dot(temp_output1 - target) - 
							(temp_output2 - target).dot(temp_output2 - target);
			gradient(i) /= 2*delta;
		}
		g = gradient;
	}
private:
	ANN nnet;
	VectorXd target;
};


void quickSort(std::vector<double>& arr, std::vector<int>& indices, int left, int right) {
	int i = left, j = right;
	double pivot = arr[(left + right) / 2];
	/*partition*/
	while (i <= j) {
		while (arr[i] < pivot)
			i++;
		while (arr[j] > pivot)
			j--;
		if (i <= j) {
			double tmp_v = arr[i];
			int tmp_i = indices[i];
			arr[i] = arr[j];
			indices[i] = indices[j];
			arr[j] = tmp_v;
			indices[j] = tmp_i;
			i++;
			j--;
		}
	}
	/*recursion*/
	if (left < j)
		quickSort(arr, indices, left, j);
	if (i < right)
		quickSort(arr, indices, i, right);
}

struct LBFGSBParam{
	int m;
	int max_iters;
	double tol;
	bool display;
	bool xhistory;
};

struct BPOutput{
	VectorXd t;
	VectorXd d;
	std::vector<int> F;
};

struct CauchyOutput{
	VectorXd xc;
	VectorXd c;
};

struct SubspaceMinOutput{
	VectorXd xbar;
	bool line_search_flag;
};


struct LBFGSB_Output {
	VectorXd x;
	double obj;
};

class LBFGSB{
public:
	LBFGSBParam param;
	LBFGSB_Output solve(ObjFunc& obj, VectorXd x0, VectorXd l, VectorXd u, LBFGSBParam params);
private:
	double f;
	VectorXd g;
	double getOptimality(VectorXd x, VectorXd g, VectorXd l, VectorXd u);
	BPOutput getBreakpoints(VectorXd x, VectorXd g, VectorXd l, VectorXd u);
	CauchyOutput getCauchyPoint(VectorXd x, VectorXd g, VectorXd l, VectorXd u,
								double theta, MatrixXd W, MatrixXd M);
	double findAlpha(VectorXd l, VectorXd u, VectorXd xc, VectorXd du, 
					std::vector<int> free_vars_idx);
	SubspaceMinOutput subspaceMin(VectorXd x, VectorXd g, VectorXd l, VectorXd u,
						VectorXd xc, VectorXd c, double theta, MatrixXd W, MatrixXd M);
	double strongWolfe(ObjFunc& obj, VectorXd x0, double f0, VectorXd g0, VectorXd p);
	double alphaZoom(ObjFunc& obj, VectorXd x0, double f0, VectorXd g0, VectorXd p,
					double alpha_lo, double alpha_hi);
};

double LBFGSB::getOptimality(VectorXd x, VectorXd g, VectorXd l, VectorXd u){
	VectorXd projected_g = x - g;
	for(int i = 0; i < x.rows(); i++) {
		if (projected_g(i) < l(i))
			projected_g(i) = l(i);
		else if (projected_g(i) > u(i))
			projected_g(i) = u(i);
	}
	projected_g = projected_g - x;
	return projected_g.cwiseAbs().maxCoeff();
}

BPOutput LBFGSB::getBreakpoints(VectorXd x, VectorXd g, VectorXd l, VectorXd u){
	int nn = x.rows();
	VectorXd t = VectorXd::Constant(nn,0.0);
	VectorXd d = -g;
	for (int i = 0; i < nn; i++) {
		if (g(i) < 0)
			t(i) = (x(i) - u(i)) / g(i);
		else if (g(i) > 0) 
			t(i) = (x(i) - l(i)) / g(i);
		else
			t(i) = DBL_MAX;
		if (t(i) < DBL_EPSILON)
			d(i) = 0.0;
	}
	// Sort elements of t and store indices in F
	std::vector<int> Fvec(nn, 0);
	std::vector<double> tvec(nn,0.0);
	for (int i = 0; i < nn; i++){
		tvec[i] = t(i);
		Fvec[i] = i;
		// std::cout << tvec[i] << " " << Fvec[i] <<std::endl;
	}
	//std::cout<<nn<<std::endl;
	// std::cout<<"BP_quicksort\n";
	quickSort(tvec, Fvec, 0, nn-1);
	// std::cout<<"BP_after_quicksort\n";

	BPOutput bpout;
	bpout.t = t;
	bpout.d = d;
	bpout.F = Fvec;
	return bpout;
}

CauchyOutput LBFGSB::getCauchyPoint(VectorXd x, VectorXd g, VectorXd l, VectorXd u,
								double theta, MatrixXd W, MatrixXd M){

	//std::cout << "BP0" << std::endl;
	BPOutput bpout = getBreakpoints(x, g, l, u);
	// std::cout << "t = " << std::endl << bpout.t << std::endl;
	// std::cout << "d = " << std::endl << bpout.d << std::endl;
	// std::cout << "F = " << std::endl;
	// for (int i = 0; i < bpout.F.size(); i++) {
	// 	std::cout << bpout.F[i] << std::endl;
	// }
	VectorXd xc = x;
	//std::cout << "BP01" << std::endl;
	VectorXd p = W.transpose() * bpout.d;
	VectorXd c = VectorXd::Constant(W.cols(), 0.0);
	double fp = - bpout.d.transpose() * bpout.d;
	double fpp = -theta * fp - p.dot(M * p);
	double fpp0 = -theta * fp;
	double dt_min = -fp / fpp;
	double t_old = 0;
	int i;
	//std::cout << "BP1" << std::endl;
	for (int j = 0; j < x.rows(); j++) {
		i = j;
		if (bpout.F[i] > 0)
			break;
	}
	int b = bpout.F[i];
	double t = bpout.t(b);
	double dt = t - t_old;
	//std::cout << "BP2" << std::endl;

	while ( (dt_min > dt) && (i < x.rows()) ) {
		if (bpout.d(b) > 0)
			xc(b) = u(b);
		else if (bpout.d(b) < 0)
			xc(b) = l(b);
		//std::cout << "BP3" << std::endl;
		double zb = xc(b) - x(b);
		c = c + dt * p;
		double gb = g(b);
		VectorXd wbt = W.row(b);
		fp += dt * fpp + gb * gb + theta * gb * zb - gb * wbt.dot(M * c);
		fpp -= theta * gb * gb - 2.0 * gb * wbt.dot(M * p) - gb * gb * wbt.dot(M * wbt);
		fpp = std::max(DBL_EPSILON * fpp0, fpp);
		p += gb * wbt;
		bpout.d(b) = 0.0;
		dt_min = -fp / fpp;
		t_old = t;
		i++;
		if (i < x.rows()){
			b = bpout.F[i];
			t = bpout.t(b);
			dt = t - t_old;
		}
	}
	// Perform final updates
	dt_min = std::max(dt_min, 0.0);
	t_old = t_old + dt_min;
	for (int j = 0; j < xc.rows(); j++) {
		int idx = bpout.F[j];
		xc(idx) += t_old * bpout.d(idx);
	}
	c += dt_min * p;

	CauchyOutput cpout;
	cpout.c = c;
	cpout.xc = xc;
	return cpout;
}

double LBFGSB::findAlpha(VectorXd l, VectorXd u, VectorXd xc, VectorXd du, 
				std::vector<int> free_vars_idx) {
	// INPUTS:
	//  l: [n,1] lower bound constraint vector.
	//  u: [n,1] upper bound constraint vector.
	//  xc: [n,1] generalized Cauchy point.
	//  du: [num_free_vars,1] solution of unconstrained minimization.
	// OUTPUTS:
	//  alpha_star: positive scaling parameter.
	double alpha_star = 1;
	int n = free_vars_idx.size();
	for (int i = 0; i < n; i++) {
		int idx = free_vars_idx[i];
		if (du(i) > 0)
			alpha_star = std::min(alpha_star, (u(idx) - xc(idx)) / du(i));
		else
			alpha_star = std::min(alpha_star, (l(idx) - xc(idx)) / du(i));
	}
	return alpha_star;
}


SubspaceMinOutput LBFGSB::subspaceMin(VectorXd x, VectorXd g, VectorXd l, VectorXd u,
								VectorXd xc, VectorXd c, double theta, MatrixXd W, MatrixXd M){
	SubspaceMinOutput subminout;
	subminout.line_search_flag = true;

	int n = x.rows();
	std::vector<int> free_vars_idx;
	std::vector<VectorXd> Z;
	for (int i = 0; i < xc.rows(); i++) {
		if ((xc(i) != u(i)) && (xc(i) != l(i))) {
			free_vars_idx.push_back(i);
			VectorXd unit = VectorXd::Constant(n,0);
			unit(i) = 1;
			Z.push_back(unit);
		}
	}

	int num_free_vars = free_vars_idx.size();
	if (num_free_vars == 0) {
		subminout.xbar = xc;
		subminout.line_search_flag = false;
		return subminout;
	}

	MatrixXd ZZ = MatrixXd::Zero(n, num_free_vars);
	for (int i = 0; i < num_free_vars; i++)
		ZZ.col(i) = Z[i];

	// compute the reduced gradient of mk restricted to free variables
	MatrixXd WTZ = W.transpose() * ZZ;
	VectorXd rr = g + theta * (xc - x) - W*M*c;
	VectorXd r = VectorXd::Constant(num_free_vars, 0.0);
	for (int i = 0; i < num_free_vars; i++)
		r(i) = rr(free_vars_idx[i]);

	// form intermediate variables
	double invtheata = 1.0 / theta;
	VectorXd v = M * WTZ * r;
	MatrixXd N = invtheata * WTZ * WTZ.transpose();
	int N_size = N.rows();
	N = MatrixXd::Identity(N_size, N_size) - M * N;
	v = N.inverse() * v;
	VectorXd du = -invtheata * r - invtheata * invtheata * WTZ.transpose()*v;

	// find alpha star
	double alpha_star = findAlpha(l, u, xc, du, free_vars_idx);
	VectorXd d_star = alpha_star * du;
	subminout.xbar = xc;
	for (int i = 0; i < num_free_vars; i++) {
		int idx = free_vars_idx[i];
		subminout.xbar(idx) += d_star(i);
	}
	return subminout;
}


double LBFGSB::strongWolfe(ObjFunc& obj, VectorXd x0, double f0, VectorXd g0, VectorXd p){
	double alpha; // return value;
	double c1 = 1e-4;
	double c2 = 0.9;
	double alpha_max = 2.5;
	double alpha_im1 = 0;
	double alpha_i = 1;
	double f_im1 = f0;
	double dphi0 = g0.dot(p);
	int i = 0;
	int max_iters = 20;
	VectorXd x;
	double f_i;
	VectorXd g_i;
	// search for alpha that satisfies strong Wolfe conditions
	while (true) {
		x = x0 + alpha_i * p;
		obj.eval(f_i, g_i, x);
		if ((f_i > f0 + c1 * dphi0) || ( (i > 1) && (f_i >= f_im1) )) {
			alpha = alphaZoom(obj, x0, f0, g0, p, alpha_im1, alpha_i);
			break;
		}
		double dphi = g_i.dot(p);
		if (fabs(dphi) <= -c2 * dphi0) {
			alpha = alpha_i;
			break;
		}
		if (dphi >= 0) {
			alpha = alphaZoom(obj, x0, f0, g0, p, alpha_i, alpha_im1);
			break;
		}

		// update
		alpha_im1 = alpha_i;
		f_im1 = f_i;
		alpha_i += 0.8 * (alpha_max - alpha_i);

		if (i > max_iters) {
			alpha = alpha_i;
			break;
		}
		i++;
	}
	return alpha;
}


double LBFGSB::alphaZoom(ObjFunc& obj, VectorXd x0, double f0, VectorXd g0, VectorXd p,
				double alpha_lo, double alpha_hi) {
	double alpha; // return value
	double c1 = 1e-4;
	double c2 = 0.9;
	int i = 0;
	int max_iters = 20;
	double dphi0 = g0.dot(p);
	double alpha_i;
	VectorXd x;
	double f_i;
	VectorXd g_i;

	while (true) {
		alpha_i = 0.5 * (alpha_lo + alpha_hi);
		alpha = alpha_i;
		x = x0 + alpha_lo * p;
		obj.eval(f_i, g_i, x);
		VectorXd x_lo = x0 + alpha_lo * p;
		double f_lo;
		VectorXd dummy_g;
		obj.eval(f_lo, dummy_g, x_lo);
		if ( (f_i > f0 + c1 * alpha_i * dphi0) || (f_i >= f_lo) )
			alpha_hi = alpha_i;
		else {
			double dphi = g_i.dot(p);
			if (fabs(dphi) <= -c2 * dphi0) {
				alpha = alpha_i;
				break;
			}
			if (dphi * (alpha_hi - alpha_lo) >= 0) {
				alpha_hi = alpha_lo;
			}
			alpha_lo = alpha_i;
		}
		i++;
		if (i > max_iters) {
			alpha = alpha_i;
			break;
		}
	}
	return alpha;
}

LBFGSB_Output LBFGSB::solve(ObjFunc& obj, VectorXd x0, VectorXd l, 
							VectorXd u, LBFGSBParam params) {
	
    int rank=0;
    rank=MPI::COMM_WORLD.Get_rank();

	int n = x0.rows();
	MatrixXd Y(n,0);
	MatrixXd S(n,0);
	MatrixXd W = MatrixXd::Zero(n,1);
	MatrixXd M = MatrixXd::Zero(1,1);
	double theta = 1;

	// initialize obj vars
	VectorXd x = x0;
	double f;
	VectorXd g;
	obj.eval(f, g, x);
	int k = 0;
	double opt;

	if (params.display && rank == 0) {
		std::cout << "iter\t\tf(x)\t\toptimality\n";
		std::cout << "-------------------------------------\n";
		opt = getOptimality(x, g, l, u);
		std::cout << k << "\t\t" << f << "\t\t" << opt << "\t\t" << std::endl;
	}

	MatrixXd xhist(n,0);
	//std::cout << "BP0" << std::endl;
	if (params.xhistory) {
		appendVec2Mat(xhist, x0);
	}

	VectorXd x_old;
	VectorXd g_old;
	VectorXd y;
	VectorXd s;
	while ((getOptimality(x, g, l, u) > params.tol) && (k < params.max_iters) ) {
		x_old = x;
		g_old = g;
		//std::cout << "BP01" << std::endl;

		CauchyOutput cpout = getCauchyPoint(x, g, l, u, theta, W, M);
		VectorXd xc = cpout.xc;
		VectorXd c = cpout.c;
		// std::cout << "xc = " << std::endl;
		// std::cout << xc << std::endl;
		// std::cout << "c = " << std::endl;
		// std::cout << c << std::endl;
		// std::cout << "BP1" << std::endl;
		SubspaceMinOutput subminout = subspaceMin(x, g, l, u, xc, c, theta, W, M);
		//std::cout << "BP2" << std::endl;

		VectorXd xbar = subminout.xbar;
		bool line_search_flag = subminout.line_search_flag;

		double alpha = 1.0;
		if (line_search_flag) {
			alpha = strongWolfe(obj, x, f, g, xbar - x);
		}
		x += alpha * (xbar - x);
		//std::cout << "BP3" << std::endl;

		// update LBFGS data structures
		obj.eval(f, g, x);
		y = g - g_old;
		s = x - x_old;
		double curv = fabs(s.dot(y));
		if (curv < DBL_EPSILON) {
			if (params.display && rank == 0) {
    			std::cout << (" warning: negative curvature detected\n");
    			std::cout << ("          skipping L-BFGS update\n");
    		}	
    		k++;
    		continue;		
		}

		if (Y.cols() < params.m) {
			appendVec2Mat(Y, y);
			appendVec2Mat(S, s);
		} else {
			Y.block(0, 0, n, params.m - 1) = Y.block(0, 1, n, params.m - 1);
			S.block(0, 0, n, params.m - 1) = S.block(0, 1, n, params.m - 1);
			Y.col(params.m - 1) = y;
			S.col(params.m - 1) = s;
		}
		//std::cout << "BP4" << std::endl;
		theta = y.dot(y) / y.dot(s);
		//std::cout << "BP401" << std::endl;
		MatrixXd temp = Y;
		//std::cout << "BP402" << std::endl;
		temp.conservativeResize(temp.rows(), temp.cols()+S.cols());
		temp.block(0, Y.cols(), S.rows(), S.cols()) = theta * S;
		//std::cout << "BP403" << std::endl;
		W = temp;
		//std::cout << W << std::endl;
		//std::cout << "BP41" << std::endl;
		MatrixXd A = S.transpose() * Y;
		MatrixXd L = getTril(A);
		MatrixXd D = -1 * A.diagonal().asDiagonal();
		int D_size = D.rows();
		int L_size = L.rows();

		//std::cout << "BP42" << std::endl;

		MatrixXd MM(D_size + L_size, D_size + L_size);
		MM.block(0, 0, D_size, D_size) = D;
		MM.block(0, D_size, D_size, L_size) = L.transpose();
		MM.block(D_size, 0, L_size, D_size) = L;
		MM.block(D_size, D_size, L_size, L_size) = theta * S.transpose() * S;
		//std::cout << "BP43" << std::endl;
		M = MM.inverse();
		//std::cout << "BP5" << std::endl;
		// update the iteration
		k++;
		if (params.xhistory) {
			appendVec2Mat(xhist, x);
		}
		if (params.display && rank == 0) {
			opt = getOptimality(x, g, l, u);
			std::cout << k << "\t\t" << f << "\t\t" << opt << "\t\t" << std::endl;
		}
	}
	if (k == params.max_iters && params.display && rank == 0) {
		std::cout << " warning: maximum number of iterations reached\n";
	}

	if (getOptimality(x,g,l,u) < params.tol && params.display && rank == 0) {
		std::cout << " stopping because convergence tolerance met!\n";
	}

	LBFGSB_Output out;
	out.x = x;
	out.obj = f;
	return out;
}


//705.023 701.723 697.852 691.804  692.91 667.241 667.139 672.044 674.512 680.647


// int main(int argc, char** argv){
// 	VectorXd x(10);
// 	x = VectorXd::Constant(10,150);
// 	VectorXd l = VectorXd::Constant(10,0.0);
// 	VectorXd u = VectorXd::Constant(10,420.0);

// 	LBFGSBParam params;
// 	params.m = 10;
// 	params.tol = 1e-2;
// 	params.max_iters = 50;
// 	params.display = true;
// 	params.xhistory = false;

// 	ObjFunc obj;
// 	obj.init();
// 	VectorXd base = VectorXd::Constant(10,673);
// 	VectorXd increment(10);
// 	increment << 40, 40, 40, 40, 40, 20, 20, 20, 20, 20;
// 	obj.setTarget(base + increment);
// 	// obj.target << 50, 30, 60, 50, 60, 70, 1, 90, 50, 50;
// 	LBFGSB opt;
// 	std::cout << "starting to solve\n";
// 	x = opt.solve(obj, x, l, u, params);
// 	std::cout << x.transpose() << std::endl;

// 	VectorXd output = obj.evalANN(x);
// 	std::cout << (output - base).transpose() << std::endl;

	
// 	// LBFGSB optimizer;
// 	// // double opt = optimizer.getOptimality(x,g,l,u);
// 	// // BPOutput bpout = optimizer.getBreakpoints(x,g,l,u);
// 	// // std::cout<<bpout.t.transpose()<<std::endl;
// 	// // std::cout<<bpout.d.transpose()<<std::endl;
// 	// // for (int i = 0; i < bpout.F.size(); i++)
// 	// // 	std::cout<<bpout.F[i]<<"  ";
// 	// // std::cout<<std::endl;
// 	// MatrixXd W = MatrixXd::Random(10,20);
// 	// MatrixXd M = MatrixXd::Identity(20,20);
// 	// VectorXd row = M.row(10);
// 	// std::cout<<row<<std::endl;

// 	// CauchyOutput cpout = optimizer.getCauchyPoint(x,g,l,u, 0.1, W, M);
// 	// std::cout<<cpout.xc.transpose()<<std::endl;
// 	// std::cout<<cpout.c.transpose()<<std::endl;
// 	// using namespace std;
// 	// MatrixXd mat(2,0);
// 	// VectorXd vec(2);
// 	// vec << 1, 1;
// 	// cout << mat.rows() << " " << mat.cols() << endl;
// 	// mat.conservativeResize(mat.rows(), mat.cols()+1);
// 	// mat.col(mat.cols()-1) = vec;
// 	// cout << mat << endl;
// 	return 0;
// }