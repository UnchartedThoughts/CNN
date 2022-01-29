package brain;

public class CNN {
    public Network[][] KERNEL;
    TrainSet[][] SET;
    TrainSet ALL;
    int SIZE;
    int KW;
    int KH;

    public CNN(int size,int kernelWidth, int kernelHeight){
        KW=kernelWidth;
        KH=kernelHeight;
        SIZE=size;
        KERNEL = new Network[size][size];
        SET = new TrainSet[size][size];
        ALL = new TrainSet(KW*KH,1);
        for(int i=0;i<size;i++){
            for(int j=0;j<size;j++){
                KERNEL[i][j] = new Network(KW*KH,(KW*KH),(KW*KH),1);
                SET[i][j] = new TrainSet(KW*KH,1);
            }
        }
    }
    
    public void add(double[][] matrix, double[] output){
        if(matrix.length!=(KW*SIZE)||matrix[0].length!=(KH*SIZE)){
            System.out.println("ERROR: matrix shape doesn't fit "+(KW*SIZE)+"x"+(KH*SIZE));
            Runtime.getRuntime().halt(0);
        }
        for(int x=0, line=0;x<matrix.length;x+=KW, line++){
            for(int y=0, col=0;y<matrix[x].length;y+=KH,col++){
                double[] input = new double[KW*KH];
                for(int i=x,count=0;i<x+KW;i++){
                    for(int j=y;j<y+KH;j++,count++){
                        input[count]=matrix[i][j];
                    }
                }
                ALL.addData(input,output);
                SET[line][col].addData(input,output);
            }
        }
    }
    
    public void train(int ep, int batch, double lr){
        if(SET[0][0].size()==0){
            System.out.println("ERROR: No data has been loaded");
            Runtime.getRuntime().halt(0);
        }
        for(int i=0;i<KERNEL.length;i++){
            for(int j=0;j<KERNEL[i].length;j++){
                KERNEL[i][j].train(SET[i][j],ep,batch,lr);
            }
        }
    }
    
    private void trainAll(int ep, int batch, double lr){
        if(SET[0][0].size()==0){
            System.out.println("ERROR: No data has been loaded");
            Runtime.getRuntime().halt(0);
        }
        for(int i=0;i<KERNEL.length;i++){
            for(int j=0;j<KERNEL[i].length;j++){
                KERNEL[i][j].train(ALL,ep,batch,lr);
            }
        }
    }
    
    public double[][] pool(double[][] matrix){
    	if(matrix.length!=(KW*SIZE)||matrix[0].length!=(KH*SIZE)){
            System.out.println("ERROR: matrix shape doesn't fit "+(KW*SIZE)+"x"+(KH*SIZE));
            Runtime.getRuntime().halt(0);
        }
        double[][] pool = new double[KERNEL.length][KERNEL[0].length];
        for(int i=0;i<KERNEL.length;i++){
            for(int j=0;j<KERNEL[i].length;j++){
                pool[i][j]=convolve(matrix,KERNEL[i][j],i,j);
            }
        }
        return pool;
    }
    
    public int[] scan(double[][] matrix, int step){
        int[] pos = new int[2];
        int istep = KW*KERNEL.length;
        int jstep = KH*KERNEL[0].length; 
        double max = 0;
        double next = 0;
        for(int x=0;x<matrix.length-(istep);x+=step){
            for(int y=0;y<matrix[x].length-(jstep);y+=step){
                double[][] input = new double[istep][jstep];
                 for(int i=x,iline=0;i<x+istep;i++,iline++){
                    for(int j=y,jcol=0;j<y+jstep;j++,jcol++){
                        input[iline][jcol]=matrix[i][j];
                    }
                }
                next=avgpool(pool(input),1,1)[0][0];
                if(max<next){
                    max=next;
                    pos[0]=x+(istep/2);
                    pos[1]=y+(jstep/2);
                }
            }
        }
        return pos;
    }
    
    public double[][] heatmap(double[][] matrix, int step){
        int istep = KW*KERNEL.length;
        int jstep = KH*KERNEL[0].length; 
        int xstep = ((matrix.length)/(step));
        int ystep = ((matrix[0].length)/(step));
        double[][] heatmap = new double[xstep][ystep];
         for(int x=0,line=0;x<matrix.length-(istep);x+=step,line++){
            for(int y=0,col=0;y<matrix[x].length-(jstep);y+=step,col++){
                double[][] input = new double[istep][jstep];
                 for(int i=x,iline=0;i<x+istep;i++,iline++){
                    for(int j=y,jcol=0;j<y+jstep;j++,jcol++){
                        input[iline][jcol]=matrix[i][j];
                    }
                }
                heatmap[line][col]=avgpool(pool(input),1,1)[0][0];
            }
        }
        
        return heatmap;
    }
    
    private double convolve(double[][] matrix, Network k, int line, int col){
        //convolution width and height(how many kernel steps)
        int cW = line*KW;
        int cH = col*KH;
        
        double[] input = new double[KW*KH];
        int count = 0;
        for(int x=cW;x<cW+KW;x++){
            for(int y=cH;y<cH+KH;y++){
               input[count++] = matrix[x][y];
            }
        }
        return k.calculate(input)[0];
    }
    
    public double[][] maxpool(double[][] matrix, int w, int h){
        int xstep = ((matrix.length)/w);
        int ystep = ((matrix[0].length)/h);
        if(xstep<=0 || ystep <=0){
            return null;
        }
        double[][] pool = new double[w][h];
        double max = 0;
        for(int x=0,line=0;x<matrix.length-(xstep-1);x+=xstep,line++){
            for(int y=0,col=0;y<matrix[x].length-(ystep-1);y+=ystep,col++){
                max=0;
                for(int i=x;i<x+xstep;i++){
                    for(int j=y;j<y+ystep;j++){
                        max=matrix[i][j]>max?matrix[i][j]:max;
                    }
                }
                pool[line][col]=max;
            }
        }
        
        return pool;
    }
    
    public double[][] avgpool(double[][] matrix, int w, int h){
        int xstep = ((matrix.length)/w);
        int ystep = ((matrix[0].length)/h);
        if(xstep<=0 || ystep <=0){
            return null;
        }
        double[][] pool = new double[w][h];
        double avg = 0;
        for(int x=0,line=0;x<matrix.length-(xstep-1);x+=xstep,line++){
            for(int y=0,col=0;y<matrix[x].length-(ystep-1);y+=ystep,col++){
                for(int i=x;i<x+xstep;i++){
                    for(int j=y;j<y+ystep;j++){
                        avg+=matrix[i][j];
                    }
                }
                pool[line][col]=avg/(xstep*ystep);
            }
        }
        
        return pool;
    }
    
    public void save(String name){
        memory(name,false);
    }
    public void load(String name){
        memory(name,true);
    }
    private void memory(String name,boolean get){
        Database db = new Database();
		if(get){
		    for(int i=0;i<SIZE;i++){
		        for(int j=0;j<SIZE;j++){
		        KERNEL[i][j].upload(db.fromFile((name+i+""+j)));
		        }
		    }
		}else{
		    for(int i=0;i<SIZE;i++){
		        for(int j=0;j<SIZE;j++){
		        db.toFile((name+i+""+j),KERNEL[i][j].download());
		        }
		    }
		}
    }
} 
