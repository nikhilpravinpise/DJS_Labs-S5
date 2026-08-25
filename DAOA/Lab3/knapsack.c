#include <stdio.h>

struct Item
{
    int weight;
    int profit;
    float ratio;
};

int main()
{
    int n, i, j, capacity;
    struct Item item[100], temp;
    float totalProfit = 0.0;

    printf("Enter the number of items: ");
    scanf("%d", &n);

    printf("Enter the profit and weight of each item:\n");
    for(i = 0; i < n; i++)
    {
        printf("Item %d\n", i + 1);
        printf("Profit: ");
        scanf("%d", &item[i].profit);
        printf("Weight: ");
        scanf("%d", &item[i].weight);

        item[i].ratio = (float)item[i].profit / item[i].weight;
    }

    printf("Enter the capacity of knapsack: ");
    scanf("%d", &capacity);

    // Sort items in descending order of profit/weight ratio
    for(i = 0; i < n - 1; i++)
    {
        for(j = i + 1; j < n; j++)
        {
            if(item[i].ratio < item[j].ratio)
            {
                temp = item[i];
                item[i] = item[j];
                item[j] = temp;
            }
        }
    }

    printf("\nSelected Items:\n");

    for(i = 0; i < n; i++)
    {
        if(capacity == 0)
            break;

        if(item[i].weight <= capacity)
        {
            printf("Take whole item (Profit = %d, Weight = %d)\n",
                   item[i].profit, item[i].weight);

            capacity -= item[i].weight;
            totalProfit += item[i].profit;
        }
        else
        {
            float fraction = (float)capacity / item[i].weight;

            printf("Take %.2f fraction of item (Profit = %d, Weight = %d)\n",
                   fraction, item[i].profit, item[i].weight);

            totalProfit += item[i].
        profit * fraction;
            capacity = 0;
        }
    }

    printf("\nMaximum Profit = %.2f\n", totalProfit);

    return 0;
}