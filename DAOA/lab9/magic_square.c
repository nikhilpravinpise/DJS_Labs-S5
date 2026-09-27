#include <stdio.h>
#include <stdlib.h>

int main()
{
    int s[3][3], i, j, k;
    int cost, minCost = 1000000;

    int magic[8][3][3] = {
        {{8, 1, 6}, {3, 5, 7}, {4, 9, 2}},
        {{6, 1, 8}, {7, 5, 3}, {2, 9, 4}},
        {{4, 9, 2}, {3, 5, 7}, {8, 1, 6}},
        {{2, 9, 4}, {7, 5, 3}, {6, 1, 8}},
        {{8, 3, 4}, {1, 5, 9}, {6, 7, 2}},
        {{4, 3, 8}, {9, 5, 1}, {2, 7, 6}},
        {{6, 7, 2}, {1, 5, 9}, {8, 3, 4}},
        {{2, 7, 6}, {9, 5, 1}, {4, 3, 8}}
    };

    printf("Enter the 3x3 matrix:\n");
    for(i = 0; i < 3; i++)
    {
        for(j = 0; j < 3; j++)
        {
            scanf("%d", &s[i][j]);
        }
    }

    for(k = 0; k < 8; k++)
    {
        cost = 0;
        for(i = 0; i < 3; i++)
        {
            for(j = 0; j < 3; j++)
            {
                cost += abs(s[i][j] - magic[k][i][j]);
            }
        }

        if(cost < minCost)
        {
            minCost = cost;
        }
    }

    printf("\nMinimum cost to form a magic square = %d\n", minCost);

    return 0;
}
