#!/usr/bin/env python
import sys, argparse


if __name__ == '__main__':

    #Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', nargs='?', help='The script to run')
    parser.add_argument('-i', action='store_true', help='Launch the interperter after executing a script')
    args = parser.parse_args()


    procId = 0
    nProcs = 1
    try:
        from mpi4py import MPI
        procId = MPI.COMM_WORLD.Get_rank()
        nProcs = MPI.COMM_WORLD.Get_size()
    except ImportError:
        print("No MPI support found, please install mpi4py if needed")
        print("otherwise just run with: python <astra-script>")
        sys.exit(-1)

    try:
        import astra
    except ImportError as exc:
        print("No ASTRA library found or loading error, please fix your installation")
        print("Exception: %s " %(exc))
        sys.exit(-1)

    #If running more than 1 process while not having built an MPI enabled ASTRA
    #then do not continue
    from astra import mpi_c       as astraMPI
    if not astraMPI.isBuiltWithMPI():
        if(nProcs > 1):
            if procId == 0:
                print("The ASTRA library is built without MPI support, but you are still trying"),
                print("to launch multiple processes.\nPlease rebuild ASTRA or modify your run parameters.")
            sys.exit(-1)

    #Split the execution. Rank 0 processes the scripts/interperter and 
    #the other ranks enter the astraMPIClientRunner loop
    if (procId != 0):
        #Load worker client script and launch the command receive loop
        from mpiLoop import astraMPIClientRunner
        c = astraMPIClientRunner()
        c.start()
    else:
        comm = MPI.COMM_WORLD
        
        astra.log.info("Master launched: rank: %d total: %d name: %s" % 
                        (comm.Get_rank(), comm.Get_size(), MPI.Get_processor_name()))

        #Process the command-line script if supplied
        if(args.script != None): 
            #execfile(args.script)
            code = compile(open(args.script).read(), args.script, 'exec')
            exec(code)
        else:
            args.i = True   #Force interpreter if nothing is supplied

        if(args.i):
            #Try to aunch IPython shell, if that fails launch default shell
            try:
                import IPython
                IPython.embed()
            except ImportError:
                print("No IPython support found. Launching default interpreter")
                
                #Launch normal shell
                import code
                import readline
                vars = globals()
                vars.update(locals()) 
                shell = code.InteractiveConsole(vars) 
                shell.interact()

        #Send exit signal to the clients
        comm.bcast(13,root=0)

    #Exit
    if(MPI.COMM_WORLD.Get_rank() == 0):
        print("Bye bye")
