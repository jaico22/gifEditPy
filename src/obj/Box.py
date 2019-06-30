class Box : 
    def __init__(self,x0,x1,y0,y1) :
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.height = y1 - y0
        self.width = x1 - x0
        
    def print(self):
        print('x0 = ' + str(self.x0) + " x1= " + str(self.x1))