using System;
using System.IO.Enumeration;

namespace Encryption_CSharp
{
    class Encryption_Module
    {
        
        private static string[,] CSV_Loader(string file)
        {
            string all_file_data = System.IO.File.ReadAllText(file);

            all_file_data = all_file_data.Replace('\n', '\r');
            string[] rows = all_file_data.Split(new char[] { '\r' }, StringSplitOptions.RemoveEmptyEntries);

            int num_rows = rows.Length;
            int num_cols = rows[0].Split(',').Length;

            string[,] processed_data = new string[num_rows, num_cols]; 

            for (int i = 0; i < num_rows; i++)
            {
                string[] a_row = rows[i].Split(',');
                for (int j = 0; j < num_cols; j++)
                {
                    processed_data[i, j] = a_row[j]; 
                }
            }

            return processed_data; 
        }

        static int Main(string[] args)
        {

            // Validate inputs 
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: encrypt.exe [INPUT CSV DATA] [OUTPUT CSV] [OUTPUT PRIVATE KEY] [OUTPUT PUBLIC KEY]");
                return 1; 
            }

            // Load file locations 
            String input_csv = args[0];
            String output_csv = args[1];
            String output_private_key = args[2];
            String output_public_key = args[3];

            // Load CSV into 2D Array 
            string [,] data = CSV_Loader(input_csv);

            // Encrypt CSV Array 

            // Write CSV Array 

            return 0; 
        }
    }
}
