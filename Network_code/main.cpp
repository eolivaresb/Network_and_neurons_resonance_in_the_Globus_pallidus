// =============================================================
// =============================================================
#include <iostream>
#include <iomanip>  // cout precision
#include <math.h>
#include "VectorMatrix.h"
#include "NervousSystem.h"
using namespace std;

//////////////////////////////////////////////////////////
///////////////// Simulation and network configuration
// Integration Parameters
const double StepSize = 0.00005;
double ttot = 3;
// Network parameters
const int N = 1000;
const int n = 10; // mean numer of synapses going out and in of a neuron
// Files for neurons and network configuration
char path_to_files[] = "./simulation_files/";
ifstream neurons_file(std::string(path_to_files) + "neurons.dat");
ifstream Connected_file(std::string(path_to_files) + "network_hubs.dat");
ifstream prc_file(std::string(path_to_files) + "diff_prc.dat");
ifstream volt_file(std::string(path_to_files) + "volt.dat");
//// delays are going to be copied from simulation_files folder to the executable folder
ifstream delays_file("delays.dat");
// Stimuli parameters
double freq = 10.0;  // Hz
double Iampl = 20; // pA
const double pi = 3.14159265359;
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int main (int argc, const char* argv[]){
/////////////////      read gmax and Erev from terminal  ////////////////////////////
    freq = atof(argv[1]);
    ttot = atof(argv[2]);
    Iampl = atof(argv[3]);
//////////////////////////////////////////////////////////
    RandomState rs;
    long randomseed = static_cast<long>(time(NULL));
    cout << "randomseed  " << randomseed << endl;
    rs.SetRandomSeed(randomseed);
    cout.precision(10);
//////////////////////////////////////////////////////////
    TVector<double> NeuronsProp(1, 2*N);
    neurons_file >> NeuronsProp;
    neurons_file.close();

    TVector<double> Connected(1, 2*n*N);
    Connected_file >> Connected;
    Connected_file.close();

    TVector<double> Delays(1, n*N);
    delays_file >> Delays;

    TVector<double> prc_params(1, 7*N);
    prc_file >> prc_params;
    prc_file.close();

///////////////// Construct connected network
    NervousSystem gp;           // Connected GPe
    gp.SetCircuitSize(N, 15*n);

    // Load mean Voltage trace
    gp.LoadVoltage(volt_file);

    // load neurons propierties
    for (int i=1; i<=N; i++){
        gp.SetNeuronW(i,NeuronsProp[2*i-1]);
        gp.SetNeuronCV(i, NeuronsProp[2*i]);
    }
    // load network connectivity and delays values
    for (int i=1; i<=n*N; i++){
        gp.SetChemicalSynapseWeight(Connected[2*i-1], Connected[2*i], 1);
        gp.Set_delay(Connected[2*i-1], Connected[2*i], Delays[i]);
    }
    // load neurons PRCs
    for (int i=1; i<=N; i++){
        for (int p=1; p<=7; p++){
            gp.PrcNeuron[i].PrcParam[p] = prc_params[7*(i-1)+p];
        }
    }
    //// Load PRC interpolation matrix
    gp.LoadPrcVectors();

//////////////////////////////////////////////
//////////////////////////////////////////////
    // Inicialization
    gp.RandomizeCircuitPhase(0.0, 1.0, rs);
    //////////////////////////////////////////////
    ////// Elements to record neurons phase densities
    int phase_bins = 1000;
    int phase_indx;
    TMatrix<int> phase_dist(1, N, 1, phase_bins);
    phase_dist.FillContents(0);
    TMatrix<double> MeanI(1, N, 1, phase_bins);
    MeanI.FillContents(0);
    int Vindx = 0;
    //////////////////////////////////////////////
    double Iext = 0;

    /////////////////// Adapting the network for 28 seconds as the neurons have rate adaptation
    gp.SetSaveSpikes(false);
    for (double t = 0; t <28; t += StepSize){
        Iext = Iampl*sin(2*pi*t*freq);
        for (int i=1; i<=N; i++)  {
            gp.externalinputs[i] = Iext;
        }
        gp.EulerStep(StepSize, t, rs);
    }
    /////////////////// Recording spikes
    gp.SetSaveSpikes(true);
    for (double t = 0; t <ttot; t += StepSize){
        Vindx = int(phase_bins * fmod(t * freq, 1.0)) + 1;
        Iext = Iampl*sin(2*pi*t*freq);
        for (int i=1; i<=N; i++)  {
            gp.externalinputs[i] = Iext;
            phase_indx = (int(phase_bins*gp.NeuronPhase(i))<0)?0:int(phase_bins*gp.NeuronPhase(i));
            phase_dist(i, 1+phase_indx)+=1;
            MeanI(i, Vindx) += gp.Conductance(i) * (gp.Erev - gp.VoltEval(gp.NeuronPhase(i)));
        }
        gp.EulerStep(StepSize, t, rs);
    }

    ///////
    ofstream phase_densities("phase_densities.dat");
    phase_densities << phase_dist;
    phase_densities.close();
    /////
    for (int i=1; i<=N; i++){
        for (int j=1; j<=phase_bins; j++){
            MeanI(i, j) /= (ttot/StepSize/phase_bins);
        }
    }
    ofstream MeanI_file("MeanI.dat");
    MeanI_file << MeanI;
    MeanI_file.close();
    ///////
}
