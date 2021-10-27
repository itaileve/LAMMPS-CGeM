##############################################################
#                                                            #
#                LAMMPS RexFF/C-GeM code                     #
#                                                            #
#                Developed by Itai Leven                     # 
#                      10/10/2020                            #
#                                                            #
#         References:                                        #
#                                                            # 
#         C-GeM: J. Phys. Chem. Lett. 2019,10,21,6820-6826   #
#  ReaxFF/C-GeM: J. Phys. Chem. Lett. 2020,11,21,9240-9247   #
#                                                            # 
#                                                            #
##############################################################


The ReaxFF/C-GeM code was developed with the 7Aug19 LAMMPS version, newer LAMMPS versions are not garanteed to work.

Instructions for compiling the LAMMPS ReaxFF/C-GeM code:

1. Download LAMMPS from https://www.lammps.org/
2. Include the ReaxFF package:
     *In the src folder "make yes-REAXFF"
1. Add files in this repository to the LAMMPS src folder
2. Compile LAMMPS, usually "make mpi" is a good choice

#######################################################################################################################

Instructions for running the ReaxFF/C-GeM test example:

1. After compilation add the lmp_mpi file (if compiled with "make mpi") to the Example folder
2. run LAMMPS with the folowing command: "./lmp_mpi < in.example"
3. check the output file dump.lammpstrj

#######################################################################################################################

Extended Lagrangian dynamics:

Idealy one would minimize the positions of the shells with respect to the cores at each timestep, however for practical purposes this process is to time consuming and therefore an extended Lagrangian aproach is adopted for integrating the shell degress of freedom. More information on extended Lagrangian molecular dynamics can be found in the following paper for Drude oscilators: J. Chem. Phys. 119, 3025 (2003)

Details of the ReaxFF/C-GeM extended Lagrangian implementation in the above example:

In the above example the shell degrees of freedom are kept at a low temperature of 1K  while the core degrees of freedom are thermostated at 300K. In the ReaxFF/C-GeM LAMMPS code this is achieved by applying different Nose-Hoover thermostates on the core and shell degrees of freedom set to the desired temperatures. In addition the extended Lagrangian formalizm requires the definition of an effective mass to the shell degrees of freedom as they are propergated on the same footing as the core degress of freedom.

This is how the thermostates are defined in the "in.example" LAMMPS input file:

1. Groups of the cores and shells are defined:
                  group cores type  1 2 4 5 6 7  #define core group
                  group shells type 3            #define shell group

2. Different thermostates are defined for each group:
                 fix             5 cores nvt temp 300 300 100          #define nvt for the cores
                 fix             6 shells nvt temp 1.0 1.0 100         #define nvt for the shells 


The mass of the shells is defined to 0.01 in the data input file "data.water_cluster" 

Masses
1   1.0080 #H
2   15.9994 #O
3   0.01 #El
4   35.453 #Cl
5   22.99  #Na
6   40.078 #Ca
7   126.90447 #I

################################################################################################################################

Instructions for calculating IR spectra

1. Go to the example_IR folder
2. Run the ReaxFF/C-GeM simulation with "./lmp_mpi < in.example_IR"
3. The simulation will output the dipole.txt file
4. run "python calc-ir-spectra.py"
5. Check output files for the IR spectrum

*The calc-ir-spectra.py code is taken from "https://github.com/EfremBraun/calc-ir-spectra-from-lammps", please cite this reference if you are using this code





