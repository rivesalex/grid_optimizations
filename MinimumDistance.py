import numpy as np
import matplotlib.pyplot as plt


class MinimumDistances:
    def __init__(self,grid_size, xlimit=1.2, ylimit=1.2):
        
        self.grid_size = grid_size
        self.xlimits = [-xlimit, xlimit]
        self.ylimits = [-ylimit, ylimit]
        
        # x and y steps
        self.x = np.linspace(-xlimit,xlimit,grid_size)
        self.y = np.linspace(-ylimit,ylimit,grid_size)
        
        # half step
        self.dx = (self.x[1]-self.x[0])/2
        self.dy = (self.y[1]-self.y[0])/2
        
        # Edges
        self.ex = np.linspace(-xlimit-self.dx, xlimit+self.dx, grid_size+1)
        self.ey = np.linspace(-ylimit-self.dy, ylimit+self.dy, grid_size+1)
        
        # Creating the meshgrid
        self.X , self.Y = np.meshgrid(self.x,self.y)
        self.XY = np.column_stack([self.X.ravel(), self.Y.ravel()])
        
        self.W = np.zeros((grid_size,grid_size))
    
    def addVector(self,V):
        if V.shape[1] != 2:
            raise AssertionError('The shape of the vector must be (x,2)')
        self.v2 = V
    
    def _getVectors(self,v1,v2):
        if type(v1) == type(None):
            v1 = self.XY
        if type(v2) == type(None):
            v2 = self.v2
        return v1,v2
    
    def vectorDistances(self,v1=None,v2=None):
        # General function to find minimum distances between vectors
        v1,v2 = self._getVectors(v1,v2)
        
        v_dif = v1[:,None,:] - v2
        distances = np.linalg.norm(v_dif,axis=2)
        return distances
    
    def minVectorDistance(self,v1=None,v2=None):
        
        v1,v2 = self._getVectors(v1,v2)
        distances = self.vectorDistances(v1,v2)
        min_distances = np.min(distances,axis=1)
        return min_distances
    
    def getMinDistIndex(self,v1=None,v2=None):
        # outputs the indexes where the difference is smallest between all v1 points to nearest v2 point
        v1,v2 = self._getVectors(v1,v2)
        distances = self.vectorDistances(v1,v2)
        index = np.where(distances == np.min(distances,axis=1).reshape(-1,1))[1]
        return index
    
    def plotV1V2(self, v1=None, v2=None):
        v1,v2 = self._getVectors(v1,v2)        
        plt.plot(v1[:,0],v1[:,1],'.')
        plt.plot(v2[:,0],v2[:,1],'.')
        plt.legend(['v1(Domain)','v2'])
    
    def plotMinimumDistances(self, v1=None, v2=None, include_maximum=True):
        v1,v2 = self._getVectors(v1,v2)
        
        self.plotV1V2(v1,v2)
        indexes = self.getMinDistIndex(v1,v2)
        minPoints = []
        for i in indexes:
            minPoints.append(v2[i])
            
        if include_maximum == True:
            plt.plot(1,0,'k--')
            
        for i,_ in enumerate(v1):
            N = np.concatenate([v1[i].reshape(1,-1),minPoints[i].reshape(1,-1)])
            plt.plot(N[:,0],N[:,1],'c--')
            
        plt.legend(['v1(domain)','v2','Maximum Minimum Distance','Min Distances'])

        if include_maximum == True:
            min_distances = self.minVectorDistance(v1,v2)
            maxMin = np.where(min_distances == max(min_distances))[0][0]
            N = np.concatenate([v1[maxMin].reshape(1,-1), minPoints[maxMin].reshape(1,-1)])
            minDistPlot = plt.plot(N[:,0],N[:,1],'k--')
        plt.grid()
        
    def plotGrid(self,XY=None):
        if type(XY) == type(None):
            XY = self.XY
        plt.plot(XY[:,0],XY[:,1],'.')
        plt.grid()
    
    def discretizeVector(self,v2):
        # This function will discretize a particular vector by doing the following:
        # Rounds it to the nearest "grid" point
        # Removes repeated values
        discretized_v2 = []
        XY = self.XY
        
        for v in v2:
            x1 = v[0]
            y1 = v[1]
            
            for i,_ in enumerate(self.ex[:-1]):
                if (x1 >= self.ex[i]) and (x1 <= self.ex[i+1]):
                    x_coord = self.x[i]
            for j,_ in enumerate(self.ey[:-1]):
                if (y1 >= self.ey[j]) and (y1 <= self.ey[j+1]):
                    y_coord = self.y[j]
            discretized_v2.append(XY[(XY[:,0] == x_coord) & (XY[:,1] == y_coord)])
            del x_coord, y_coord
            
        discretized_v2 = np.vstack(discretized_v2)
        self.v2 = np.unique(discretized_v2,axis=0)
        return self.v2

    def _reduceGridComplexity_Odd(self,XY=None,output=False):
        if type(XY) == type(None):
            XY = self.XY
        out = []
        for i in range(XY.shape[0]):
            if i%2==0:
                out.append(i)
                continue
        out = np.array(out)
        self.XY_reduced = XY[out]
        if output == True:
            return XY[out]
        
    def _reduceGridComplexity_Even(self,XY=None,output=False):
        if type(XY) == type(None):
            XY = self.XY
        out = []
        n = int(np.sqrt(XY.shape[0]))
        switch = False

        for i in range(XY.shape[0]):
            if switch == False:
                if i%2==0:
                    out.append(i)
            if switch == True:
                if i%2 != 0:
                    out.append(i)
            if (i+1) % n == 0:
                switch = (switch == False)
        out = np.array(out)
        self.XY_reduced = XY[out]
        if output == True:
            return XY[out]
    
    def reduceGridComplexity(self,XY=None, grid_type='mesh'):
        if type(XY) == type(None):
            XY = self.XY
        is_odd = (int(np.sqrt(XY.shape[0])) % 2) != 0
        
        if grid_type == 'mesh':
            if is_odd:
                return self._reduceGridComplexity_Odd(XY,output=True)
            return self._reduceGridComplexity_Even(XY,output=True)
        
        if grid_type == 'vertical':
            if is_odd:
                return self._reduceGridComplexity_Even(XY,output=True)
            return self._reduceGridComplexity_Odd(XY,output=True)
            
        
        
    
            
 
        
        