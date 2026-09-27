#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX 20

int G[MAX][MAX];
int x[MAX];
int n, m;
bool found = false;

bool isSafe(int k, int c){
    for (int i = 0; i < n; i++){
        if (G[k][i] == 1 && c == x[i]){
            return false;
        }
    }
    return true;
}

void graphColor(int k){
    for (int c = 1; c <= m && !found; c++){
        if (isSafe(k, c)){
            x[k] = c;
            if (k + 1 < n)
                graphColor(k + 1);
            else{
                found = true;
                return;
            }
        }
    }
}

int main(){
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    printf("Enter number of colours: ");
    scanf("%d", &m);

    printf("Enter the adjacency matrix:\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            scanf("%d", &G[i][j]);
        }
    }

    printf("The adjacency matrix is:\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%d ", G[i][j]);
        }
        printf("\n");
    }

    graphColor(0);

    if (found){
        printf("The graph colouring is: ");
        for(int i = 0; i < n; i++){
            printf("%d ", x[i]);
        }
        printf("\n");
    } else {
        printf("No colouring possible with %d colours.\n", m);
    }

    return 0;
}
