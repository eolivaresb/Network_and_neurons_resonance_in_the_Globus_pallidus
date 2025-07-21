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
double noise_level = 0.0; // std noise pulses in pA
double pulse_width = 0.0005; // 0.5 ms pulse width
// Network parameters
const int N = 1000;
const int n = 10; // mean numer of synapses going out and in of a neuron
// Files for neurons and network configuration
char path_to_files[] = "../../../simulation_files/";
ifstream neurons_file(std::string(path_to_files) + "neurons.dat");
ifstream Connected_file(std::string(path_to_files) + "network_hubs.dat");
ifstream prc_file(std::string(path_to_files) + "diff_prc.dat");
ifstream volt_file(std::string(path_to_files) + "volt.dat");
//// delays is going to be copied from simulation_files folder to the executable folder
ifstream delays_file("delays.dat");
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int main (int argc, const char* argv[]){
/////////////////      read gmax and Erev from terminal  ////////////////////////////
    ttot = atof(argv[1]);
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
    ofstream noise("noise.dat");
    //////////////////////////////////////////////
    double Iext = 0;
    /////////////////// Adapting the network for 28 seconds as the neurons have rate adaptation
    gp.SetSaveSpikes(false);
    for (double t = 0; t <28.; t += StepSize){
        gp.EulerStep(StepSize, t, rs);
    }
    /////////////////// Recording spikes
    gp.SetSaveSpikes(true);
    int Steps_per_pulse = (int)(pulse_width / StepSize);
    int nindx = (int)(Steps_per_pulse/2);
    for (double t = 0; t <ttot; t += StepSize){
        if (nindx == 10){
            Iext = rs.GaussianRandom(0.0, pow(noise_level, 2)); // 40 pA std
            noise << Iext << endl;
            nindx = 0;
        }
        for (int i=1; i<=N; i++)  {
            gp.externalinputs[i] = Iext;
        }
        gp.EulerStep(StepSize, t, rs);
        nindx +=1;
    }
    ///////
    noise.close();
    ///////
}
