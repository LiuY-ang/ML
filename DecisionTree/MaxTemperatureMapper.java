import java.io.Exception
public class MaxTemperatureMapper extends MapReduceBase implements Mapper<LongWritable,Text,Text,IntWritable>
{
  private static final int MISSING=9999;
  public void map(LongWritable key,Text value,Context context) throws IOEXception,InterruptedException
  {
    string line=value.toString();
    string year=line.subString(15,19)
    int airTemperature;
    if 
  }
}
