// ************************************************************
// A neural network of phase model neurons
// (based on the  Randall Beer CTRNN class)
//
// EOlivares
//  11/19 Created
// ************************************************************
#include "NervousSystem.h"
#include "random.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
using namespace std;
#include <iomanip>  // cout precision

/// Global variables, Volt(phase)
TVector<double> xvolt(1, 1001);
TVector<double> yvolt(1, 1001);

//// File to store simulation results
ofstream spkt("Spikes_times.dat", ios::out|ios::binary);
ofstream spki("Spiking_neurons.dat");

// Constructors and Destructors
// ****************************
// The constructor
NervousSystem::NervousSystem(int newsize, int newmaxchemconns)
{
    SetCircuitSize(newsize, newmaxchemconns);
}

// The destructor
NervousSystem::~NervousSystem()
{
    SetCircuitSize(0, 0);
}

// *********
// Utilities
// *********

void NervousSystem::SetCircuitSize(int newsize, int newmaxchemconns)
{
    ////// Connections
    size = newsize;
    if (newmaxchemconns == -1) maxchemconns = size;
    else maxchemconns = min(newmaxchemconns, size);

    NumChemicalConns.SetBounds(1,size);
    for (int i = 1; i <= size; i++)
        NumChemicalConns[i] = 0;
    chemicalweights.SetBounds(1,size,1,maxchemconns);

    ////// Phase variables
    Phases.SetBounds(1,size);
    Phases.FillContents(0.0);

    Phases_pre.SetBounds(1,size);
    Phases_pre.FillContents(0.0);

    ////// Neuron Propierties
    W.SetBounds(1,size);
    W.FillContents(10.0); // Default: 10 Hz
    CV.SetBounds(1,size);
    CV.FillContents(0.0); // Default: No noise

    ////// rate adaptation variables
    W_adapt.SetBounds(1,size);
    W_adapt.FillContents(0.0);
    afactor = 0.5;  // adaptation equilibrium
    tau_adapt = 5;  // 5 seconds, this could be heterogenous

    ////// Synapse propierties
    Erev = -0.07; // default to -73 mV
    IpscRef = 2500; // 2.5 nS average including failures measured by Matthew Higgs
    IpscStd = 1750; // 1.75 nS Standar deviation of a normal distribution
    tauG = 0.0024; // 2.4 ms, aprox for 0.3 taur, 2.1 taud measured in GPe Parv by Matthew Higgs

    //////Conductances variables
    Conductance.SetBounds(1,size);
    Conductance.FillContents(0.0);

    Conductance_pre.SetBounds(1,size);
    Conductance_pre.FillContents(0.0);

    //////Conductances Buffer
    BufferSize = 1000; // should be big enough as to store a delay without wraparound (max(delay)/stepsize)
    delay.SetBounds(1,size, 1, size); // This will be a very sparse matrix, could be optimezed
    delay.FillContents(0.00); // Default: no delay
    ConductanceBuffer.SetBounds(1,size,1,BufferSize);
    ConductanceBuffer.FillContents(0.0);

    //// Excittory synapses
    Erev_Glu = 0.000; //
    tauG_Glu = 0.002; //
    EpscRef = 1500; // 0.75 nS
    Conductance_Glu.SetBounds(1,size);
    Conductance_Glu.FillContents(0.0);

    Rates_Glu.SetBounds(1,size);
    Rates_Glu.FillContents(0.0);


    // External inputs
    externalinputs.SetBounds(1,size);
    externalinputs.FillContents(0.0);

    /// PRC parametrization
    //////PRCs units: cycles/(pA*s)
    PRCsize = 7;
    PrcNeuron.SetBounds(1, size);
    for (int i=1; i<=size; i++) // Load default PRC
    {
        PrcNeuron[i].PrcParam.SetBounds(1, PRCsize);
        PrcNeuron[i].PrcParam[1] = 0.469;
        PrcNeuron[i].PrcParam[2] = 0.212;
        PrcNeuron[i].PrcParam[3] = 0.124;
        PrcNeuron[i].PrcParam[4] = 0.608;
        PrcNeuron[i].PrcParam[5] = 3.650;
        PrcNeuron[i].PrcParam[6] = 28.60;
        PrcNeuron[i].PrcParam[7] = 0.980;
    }

    /// PRC Vector to evaluation
    prc_vectors.SetBounds(1, size);
    prcbins = 2001;  // This number of bins provide a good enough aproximation to the countinous PRC evaluation
    for (int i=1; i<=size; i++)
    {
      prc_vectors(i).SetBounds(1, prcbins);
    }

    ///// Should be recalculated each time PRCs are changed. Not relevant if noiseless neurons
    Sensitivity.SetBounds(1,size);
    Sensitivity.FillContents(0.0);
    for (int i=1; i<=size; i++)
    {
        for (double p = 0; p<=1; p+=0.0001){
            Sensitivity[i] += pow(PrcEval(p, PrcNeuron[i].PrcParam), 2);
        }
        Sensitivity[i] /= 10001;
    }

    SaveSpikes = true; // Defauls save spikes to a file
}

// *********
// Accessors
// *********

double NervousSystem::ChemicalSynapseWeight(int from, int to)
{
    for (int i = 1; i <= NumChemicalConns(from); i++) {
        if (chemicalweights[from][i].to == to)
            return chemicalweights[from][i].weight;
    }
    return 0.0;
}

void NervousSystem::SetChemicalSynapseWeight(int from, int to, double value)
{
    // If the connection is already stored, just change its value
    for (int i = 1; i <= NumChemicalConns[from]; i++)
        if (chemicalweights[from][i].to == to) {
            chemicalweights[from][i].weight = value;
            return;
        };
    // Otherwise, make sure we have room for an additional connection ...
    if (NumChemicalConns[from] == maxchemconns) {
        cerr << "Maximum chemical synapses (" << maxchemconns << ") exceeded from neuron " << from << endl;
        exit(EXIT_FAILURE);
    }
    // ... and store it
    int i = ++NumChemicalConns[from];
    chemicalweights[from][i].to = to;
    chemicalweights[from][i].weight = value;
}

//

void NervousSystem::LoadVoltage(ifstream &ifs)
{
    TVector<double> volt(1, 2002);
    ifs >> volt;
    for (int i=1; i<=1001; i++){
        xvolt[i] = volt[2*i -1];
        yvolt[i] = volt[2*i];
    }
}

double NervousSystem::VoltEval(double x)
{
    int i = 1+ int(x*1000);    // Left interpolation edge
    double xL = xvolt[i], yL = yvolt[i], xR = xvolt[i+1], yR = yvolt[i+1];// points on either side
    double dydx = ( yR - yL ) / ( xR - xL );                             // gradient
    return yL + dydx * ( x - xL );                                       // linear interpolation
}
void NervousSystem::LoadPrcVectors()
{
  for (int i=1; i<=size; i++)
  {
      for (int p = 1; p<=prcbins; p++)
      {
          prc_vectors(i)(p) = PrcEval((p-1.)/(prcbins-1.), PrcNeuron(i).PrcParam);
      }
  }
}

double NervousSystem::PrcEvalInterp(int neuron, double x)
{
  int p = 1+ int(x*prcbins); // From phase (x) to indx position in PRC evaluated vector (p)
  if ((p<=0)||(p>=prcbins)){
    return 0.0;
  }
  else{
    return prc_vectors(neuron)(p);
  }
}
// *******
// Control
// *******

// Randomize the neuron phases in the network.

void NervousSystem::RandomizeCircuitPhase(double lb, double ub)
{
    for (int i = 1; i <= size; i++)
        SetNeuronPhase(i, UniformRandom(lb, ub));
}

void NervousSystem::RandomizeCircuitPhase(double lb, double ub, RandomState &rs)
{
    for (int i = 1; i <= size; i++)
        SetNeuronPhase(i, rs.UniformRandom(lb, ub));
}

// Integrate a circuit one step using Euler integration.
void NervousSystem::EulerStep(double stepsize, double t, RandomState &rs)
{
    int post; // variable to store postsynaptic neuron index
    // Manage delay index to store conductance changes on ConductanceBuffer matrix
    static int DelayIndx = 0;
    DelayIndx = DelayIndx % BufferSize;
    DelayIndx ++;
    //////////////////
    double event = 0; // variable to store IPSG amplitude
    double tspk;

    // backup current conductance and phases
    for (int i = 1; i <= size; i++) {
        Phases_pre[i] = Phases[i];
        Conductance_pre[i] = Conductance[i];
    }

    // Update network states (phases, conductances, save spikes).
    for (int i = 1; i <= size; i++) { // sweep over presynaptic neurons
        // External input
        double input = externalinputs[i];
        // noise input to achieve a CV = CV[i]
        double noise = rs.GaussianRandom(0.0, pow(CV[i],2)*W[i]/(Sensitivity[i] * stepsize));
        // Total input (external + synaptic + noise) // using non updated conductances
        input += Conductance_pre[i] * (Erev - VoltEval(Phases[i])) + noise;
        input += Conductance_Glu[i] * (Erev_Glu - VoltEval(Phases[i]));
        // Euler step on W_adapt
        W_adapt[i] -= stepsize/tau_adapt*( W_adapt[i] + afactor*input * PrcEvalInterp(i, Phases(i)));
        //Euler step on phase
        Phases[i] += stepsize*(W[i] + W_adapt[i] + input * PrcEvalInterp(i, Phases(i)));
        // If phase >=1 then: update conductance in postsynaptic neurons, store spike, update phase
        if (Phases[i] >=1.0){ // neurons i fires (presynaptic = i)
            // Neuron i fires, Update all postsynaptic conductances
            for (int j = 1; j <= NumChemicalConns[i]; j++){ // update all postsynaptic conductances
                post = chemicalweights[i][j].to;
                event = rs.GaussianRandom(IpscRef, pow(IpscStd, 2));
                if (event > 0){
                    // Update postsynaptic conductance on the ConductanceBuffer array with a delay defined in delay matrix
                    ConductanceBuffer[post][1 + (DelayIndx + int(delay[i][post] / stepsize) ) % BufferSize] += event;
                }
            }
            if (SaveSpikes){//// Store spike
                tspk = t + stepsize *(1-Phases_pre[i])/(Phases[i]-Phases_pre[i]);
                spkt.write((char*)&tspk, sizeof(double));
                spki << i << endl;
            }
            //// Reset Phase
            Phases[i] -= 1.0;
        }
        if (Phases[i] < 0) { Phases[i] = 0.0;}
        // Update conductances
        ////// IPSP
        Conductance[i] -= stepsize*(Conductance[i]/tauG);
        Conductance[i] += ConductanceBuffer[i][DelayIndx];
        ConductanceBuffer[i][DelayIndx] = 0.0;

        ///// EPSP
        Conductance_Glu[i] -= stepsize*(Conductance_Glu[i]/tauG_Glu);
        // Update Excitatory synapses events
        if (rs.UniformRandom(0.0,1.0) > exp(-stepsize * Rates_Glu[i]) ){
            Conductance_Glu[i] += EpscRef;
        }
    }
}
