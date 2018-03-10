#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

double *outputarray;
double etta = 0.01;

typedef struct input{
    double value;
    double *weight;

}Input;

typedef struct neuron{
    // int number;
    double error;
    double net;
    struct neuron *next;
    Input from;

}Neuron;


void printout(Neuron *neuron, int neurons){
    while(neuron!=NULL){
        printf("%lf\n", neuron->from.value);
        // printf("%lf\n", neuron->net);
        // printf("%lf\n\n", neuron->error);
        neuron = neuron->next;
    }
}

void print(Neuron *neuron, int neurons){
    int j=0;
    while(neuron!=NULL){
        // printf("%lf\n", neuron->from.value);
        // printf("%lf\n", neuron->net);
        for(j=0;j<neurons;j++){
            printf("%lf", neuron->from.weight[j]);
        }
        printf("\n");
        // printf("%lf\n\n", neuron->error);

        neuron = neuron->next;
    }
}

double activationfunction(double value){
    double out = pow(2.71828, value);
    out = out /(1.0+out);
    return out; 
}

void InputLayer(Neuron *temp, int input, int neurons){
    int i=0, j=0;
    for(i=0;i<input;i++){
        Neuron *temp1 = (Neuron*)malloc(sizeof(Neuron));
        temp1->from.weight = (double*)malloc(neurons*sizeof(double));
        temp1->from.value = 0;
        temp1->error = 0;
        temp1->net=0;
        temp1->next = NULL;

        for(j=0;j<neurons;j++){
            temp1->from.weight[j] = ((((double)rand()/RAND_MAX)*2.0)-1.0);
        }
        temp->next = temp1;
        temp = temp->next;
    }
}

void hiddenlayer(Neuron *hidden, int neurons, int classes){
    int i=0,j=0;
    
    for(i=0;i<neurons;i++){
        Neuron *hiddentemp = (Neuron*)malloc(sizeof(Neuron));
        hiddentemp->from.weight = (double*)malloc(classes*sizeof(double));
        hiddentemp->from.value = 0;
        for(j=0;j<classes;j++){
            hiddentemp->from.weight[j] = ((((double)rand()/RAND_MAX)*2.0)-1.0);
        }
        hiddentemp->net = 0;
        hiddentemp->error = 0;
        hiddentemp->next=NULL;
        hidden->next = hiddentemp;
        hidden = hidden->next;
    }
}

void outputclasslayer(Neuron *output, int classes){
    int i=0,j=0;

    for(i=0;i<classes;i++){
        Neuron *outputtemp = (Neuron*)malloc(sizeof(Neuron));
        outputtemp->from.value = 0;
        outputtemp->net = 0;
        outputtemp->error = 0;
        outputtemp->next=NULL;

        output->next = outputtemp;
        output = output->next;
    }
}

void PassHiddenLayer(Neuron *hidden1, Neuron *input1){
    Neuron *temp = input1;
    int i=0;
    double act=0;

    hidden1 = hidden1->next;
    while(hidden1!=NULL){
        while(input1!=NULL){
            act += input1->from.weight[i] * input1->from.value;
            input1 = input1->next;
        }
        hidden1->net = act;
        hidden1->from.value = activationfunction(act);
        input1 = temp;
        act=0; i++;
        hidden1 = hidden1->next;
    }
}

void PassOutputLayer(Neuron *classes, Neuron *hidden){
    Neuron *temp = hidden;
    int i=0;
    double act=0;

    classes = classes->next;
    while(classes!=NULL){
        while(hidden!=NULL){
            act += hidden->from.weight[i] * hidden->from.value;
            hidden = hidden->next;
        }
        classes->net = act;
        classes->from.value = activationfunction(act);
        hidden = temp;
        i++; act=0;
        classes = classes->next;
    }
}

void ErrorCal(Neuron *outputneuron, int i){
    outputneuron = outputneuron->next;
    int j=1;
    while(outputneuron!=NULL){
        if(j==outputarray[i]){
            outputneuron->error = (1 - outputneuron->from.value)*(activationfunction(outputneuron->net)*(1-activationfunction(outputneuron->net)));
        }
        else{
            outputneuron->error = (0 - outputneuron->from.value)*(activationfunction(outputneuron->net)*(1-activationfunction(outputneuron->net)));
        }
        outputneuron = outputneuron->next;
        j++;
    }
}

void HiddenLayerError(Neuron *hidden, Neuron *outputclasses){
    double error = 0;
    int i=0;

    Neuron *tempout = outputclasses->next;
    while(hidden!=NULL){
        while(tempout!=NULL){
            error += tempout->error*hidden->from.weight[i]*((activationfunction(hidden->net)*(1-activationfunction(hidden->net))));
            tempout = tempout->next;
            i++;
        }
        hidden->error = error;
        hidden=hidden->next;
        i=0; error=0;
        tempout = outputclasses->next;
    }
}



void AdjustHiddenLayerWeight(Neuron *hidden, Neuron *classes){
    int i=0;

    Neuron *tempclasses = classes->next;
    while(hidden!=NULL){
        while(tempclasses!=NULL){
            hidden->from.weight[i] += etta * tempclasses->error * hidden->from.value;
            tempclasses = tempclasses->next;
            i++;
        }
        tempclasses = classes->next;
        hidden = hidden->next;
        i=0;
    }
}

void AdjustInputLayerWeight(Neuron *inputlayer, Neuron *hidden){
    int i=0;
    
    Neuron *temphidden = hidden->next;
    while(inputlayer!=NULL){
        while(temphidden!=NULL){
            inputlayer->from.weight[i] += etta * temphidden->error * inputlayer->from.value;
            temphidden = temphidden->next;
            i++;
        }
        temphidden = hidden->next;
        inputlayer = inputlayer->next;
        i=0;
    }
}

void FindClass(Neuron *outputlayer, FILE *outtrain){
    double max=0;
    int cls=0, j=0;

    outputlayer=outputlayer->next;
    while(outputlayer!=NULL){
        if(max<outputlayer->from.value){
            max = outputlayer->from.value;
            cls=j;
        }
        outputlayer=outputlayer->next;
        j++;
    }
    fprintf(outtrain, "%d\n", cls+1);
}

void TrainNetwork(Neuron *neuron, Neuron *hidden, Neuron *outputneuron, int neurons){
    Neuron *temp = neuron->next;
    int i=0,in,j=0,flag=0;
    
    printf("Training .... .\n");
    while(j<1000){
        FILE *fp = fopen("traindata.txt", "r");
        for(i=0;i<2216;i++){
            while(temp!=NULL){
                fscanf(fp, "%d", &in);
                temp->from.value = in;
                temp = temp->next;
            }
            
            PassHiddenLayer(hidden, neuron);
            
            PassOutputLayer(outputneuron, hidden);
            
            ErrorCal(outputneuron, i);
            
            HiddenLayerError(hidden, outputneuron);
            
            AdjustHiddenLayerWeight(hidden, outputneuron);
            
            AdjustInputLayerWeight(neuron, hidden);

            temp = neuron->next;
        }
        j++;  fclose(fp);
    }
    // printf("%d %d\n", j, i);
    printf("Done.\n");
}

void TestNetwork(Neuron *neuron, Neuron *hidden, Neuron *outputneuron, int neurons, FILE *outtrain){
    int i=0,in,j=0;
    FILE *fp = fopen("testdata.txt", "r");

    Neuron *temp = neuron->next;
    for(i=0;i<998;i++){
        while(temp!=NULL){
            fscanf(fp, "%d", &in);
            temp->from.value = in;
            temp = temp->next;
        }
        
        PassHiddenLayer(hidden, neuron);
        
        PassOutputLayer(outputneuron, hidden);
        
        FindClass(outputneuron, outtrain);
        
        temp = neuron->next;
    }
    fclose(fp);
}

void accuracy(){
   FILE *f1 = fopen("testoutput.txt", "r");
   FILE *f2 = fopen("testclass.txt", "r");
    int i=0, count=0,a,b;
    while(i<998){
        fscanf(f1, "%d", &a);
        fscanf(f2, "%d", &b);
        // printf("%d %d\n", a,b);
        if(a==b){
            count++;
        }
        i++;
    }
    // printf("%d\n", count);
    double acc = (count/998.0)*100;
    printf("Accuracy : %lf\n", acc);
}

int main(){
    srand ( time ( NULL));
    int input,neurons,classes,i=0,j,in;
    
    input = 16;
    neurons = rand() % 4 + 5;
    classes = 10;
    printf("%d %d %d\n", input, neurons, classes);

    outputarray = (double*)malloc(2217*sizeof(double));
    FILE *fp = fopen("trainclass.txt", "r");
    while(fscanf(fp, "%d", &in) != EOF){
        outputarray[i] = in;
        i++;
    }
    fclose(fp);

    //creating input layer
    Neuron *neuron = (Neuron*)malloc(sizeof(Neuron));
    neuron->from.weight = (double*)malloc(neurons*sizeof(double));
    for(j=0;j<neurons;j++)
        neuron->from.weight[j] = ((((double)rand()/RAND_MAX)*2.0)-1.0);
    neuron->from.value = 1;
    neuron->error = 0;
    neuron->next = NULL;
    InputLayer(neuron, input, neurons);
    
    //Initialize Hidden Layer
    Neuron *hidden = (Neuron*)malloc(sizeof(Neuron));
    hidden->from.weight = (double*)malloc(classes*sizeof(double));
    for(j=0;j<classes;j++)
            hidden->from.weight[j] = ((((double)rand()/RAND_MAX)*2.0)-1.0);
    hidden->from.value = 1;
    hidden->error = 0;
    hidden->next = NULL;
    hiddenlayer(hidden, neurons, classes);

    //Initialize Output layer
    Neuron *outputneuron = (Neuron*)malloc(sizeof(Neuron));
    outputneuron->from.value = 0;
    outputneuron->error = 0;
    outputneuron->next = NULL;
    outputclasslayer(outputneuron, classes);

    //Start Train neural network
    TrainNetwork(neuron, hidden, outputneuron, neurons);
    
    //Test neural network
    FILE *outtrain = fopen("testoutput.txt", "w");
    TestNetwork(neuron, hidden, outputneuron, neurons, outtrain);
    
    fclose(outtrain);
    printf("Attributes in input :%d\nNumber of neurons in Hidden layer :%d\nClasses :%d\n", input, neurons, classes);
    
    accuracy();
}