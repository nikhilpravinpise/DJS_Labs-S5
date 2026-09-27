#include <stdio.h>
#include <stdlib.h>

int board[25];      // board[i] = column of the queen in row i
int count = 0;

// checks whether a queen can be placed at (row, column)
int place(int row, int column)
{
    int i;
    for(i = 1; i <= row - 1; i++)
    {
        // check same column and same diagonal
        if(board[i] == column)
            return 0;
        else if(abs(board[i] - column) == abs(i - row))
            return 0;
    }
    return 1;
}

// prints one solution as a chessboard
void print_board(int n)
{
    int i, j;
    count++;
    printf("\nSolution %d:\n", count);
    for(i = 1; i <= n; i++)
    {
        for(j = 1; j <= n; j++)
        {
            if(board[i] == j)
                printf(" Q ");
            else
                printf(" - ");
        }
        printf("\n");
    }
}

void queen(int row, int n)
{
    int column;
    for(column = 1; column <= n; column++)
    {
        if(place(row, column))
        {
            board[row] = column;    // no conflict, place queen
            if(row == n)
                print_board(n);     // all queens placed
            else
                queen(row + 1, n);  // try next row
        }
    }
}

int main()
{
    int n;

    printf("Enter number of queens: ");
    scanf("%d", &n);

    if(n < 1 || n > 20)
    {
        printf("Please enter n between 1 and 20\n");
        return 0;
    }

    queen(1, n);

    if(count == 0)
        printf("\nNo solution exists for %d queens.\n", n);
    else
        printf("\nTotal solutions found = %d\n", count);

    return 0;
}
