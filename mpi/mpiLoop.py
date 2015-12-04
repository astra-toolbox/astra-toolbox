
from mpi4py import MPI
from astra import data3d_c    as data3d
from astra import algorithm_c as algorithm
from astra import projector_c as projector
from astra import mpi_c       as astraMPI
from astra import log_c       as astraLog
import astra

class astraMPIClient_Data3D:
    
    def __init__(self):
        # Python-level objects and code
        self.comm  = MPI.COMM_WORLD
        self.size  = self.comm.Get_size()
        self.rank  = self.comm.Get_rank()

    def process(self, cmd):
        if   cmd == 101:
            data3d.create(None, None, None)
        elif cmd == 102:
            data3d.get(None)
        elif cmd == 103:
            data3d.delete(None)
        elif cmd == 104:
            data3d.clear()
        elif cmd == 105:
            data3d.store(None, None)
        elif cmd == 106:
            data3d.dimensions_global_volume_geometry(None)
        elif cmd == 107:
            data3d.dimensions_global(None)
        elif cmd == 108:
            data3d.sync(None)
        elif cmd == 109:
            data3d.change_geometry(None, None)
        else:
            print("Not implemented data3d command")

class astraMPIClient_Projector:

    def __init__(self):
        # Python-level objects and code
        self.comm  = MPI.COMM_WORLD
        self.size  = self.comm.Get_size()
        self.rank  = self.comm.Get_rank()

    def process(self, cmd):
        if   cmd == 301:
            projector.create(None)
        elif cmd == 303:
            projector.delete(None)
        elif cmd == 304:
            projector.clear()
        else:
            print("Not implemented projection command")


class astraMPIClient_Algorithm:
    def __init__(self):
        # Python-level objects and code
        self.comm  = MPI.COMM_WORLD
        self.size  = self.comm.Get_size()
        self.rank  = self.comm.Get_rank()

    def process(self, cmd):
        if   cmd == 201:
            algorithm.create(None)
        elif cmd == 202:
            algorithm.run(None, None)
        elif cmd == 203:
            algorithm.delete(None)
        elif cmd == 204:
            algorithm.clear()
        else:
            print("Not implemented algorithm command")

class astraMPIClient_MPI:
    
    def __init__(self):
        # Python-level objects and code
        self.comm  = MPI.COMM_WORLD
        self.size  = self.comm.Get_size()
        self.rank  = self.comm.Get_rank()

    def process(self, cmd):
        if   cmd == 401:
            astraMPI.create(None,None)
        elif cmd == 402:
            astraMPI._runInternal(None, None)
        else:
            print("Not implemented astraMPI command")

class astraMPIClient_LOG:
    def __init__(self):
        # Python-level objects and code
        self.comm  = MPI.COMM_WORLD
        self.size  = self.comm.Get_size()
        self.rank  = self.comm.Get_rank()

    def process(self, cmd):
        if   cmd == 501:
            astraLog.log_enable()
        elif cmd == 502:
            astraLog.log_enableScreen()
        elif cmd == 503:
            astraLog.log_enableFile()
        elif cmd == 504:
            astraLog.log_disable()
        elif cmd == 505:
            astraLog.log_disableScreen()
        elif cmd == 506:
            astraLog.log_disableFile()
        elif cmd == 507:
            astraLog.log_setFormatFile(None)
        elif cmd == 508:
            astraLog.log_setFormatScreen(None)
        elif cmd == 509:
            astraLog.log_setOutputScreen(None, None)
        elif cmd == 510:
            astraLog.log_setOutputFile(None, None)
        else:
            print("Not implemented astraMPI command")



class astraMPIClientRunner:

    def __init__(self):
        # Python-level objects and code
        self.comm  = MPI.COMM_WORLD
        self.size  = self.comm.Get_size()
        self.rank  = self.comm.Get_rank()
        self.pname = MPI.Get_processor_name()

        astra.log.info("Client launched: rank: %d total: %d name: %s" % 
                        (self.rank, self.size, self.pname))

    def start(self):
        clientD3D = astraMPIClient_Data3D()
        clientAlg = astraMPIClient_Algorithm()
        clientPrj = astraMPIClient_Projector()
        clientMPI = astraMPIClient_MPI()
        clientLog = astraMPIClient_LOG()

        while True:
            data = None
            recvInfo = self.comm.bcast(data, root=0)

            if(recvInfo == 13): 
                print("Client Python script exiting")
                break
            elif recvInfo >= 100 and recvInfo < 200:            
                clientD3D.process(recvInfo)
            elif recvInfo >= 200 and recvInfo < 300:            
                clientAlg.process(recvInfo)
            elif recvInfo >= 300 and recvInfo < 400:            
                clientPrj.process(recvInfo)
            elif recvInfo >= 400 and recvInfo < 500:            
                clientMPI.process(recvInfo)
            elif recvInfo >= 500 and recvInfo < 600:
                clientLog.process(recvInfo)
            else:
                print("Invalid MPI command: ", recvInfo)

            #print "Received: ", recvInfo, " waiting for more :D", self.rank



if __name__ == '__main__':
        c = astraMPIClientRunner()
        c.start()
