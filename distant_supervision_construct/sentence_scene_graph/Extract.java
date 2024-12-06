import edu.stanford.nlp.scenegraph.AbstractSceneGraphParser;
import edu.stanford.nlp.scenegraph.RuleBasedParser;
import edu.stanford.nlp.scenegraph.SceneGraph;
// import javafx.util.Pair;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

class Worker extends Thread {
    ArrayList<JSONObject> task;
    int index;
    String outFolder;

    public Worker(ArrayList<JSONObject> task, int index, String outFolder) {
        this.task = task;
        this.index = index;
        this.outFolder = outFolder;
    }

    static void writeFile(String file, ArrayList<String> lines) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter(file, "utf-8");
        for (String line : lines){
            writer.println(line);
        }
        writer.close();
    }

    public void run()
    {
        long start = System.currentTimeMillis();
        AbstractSceneGraphParser parser = new RuleBasedParser();

        ArrayList<String> lines = new ArrayList<>();
        for (int i = 0; i < task.size(); i++) {

            // coco
            JSONObject o = (JSONObject) task.get(i);
            String sentence = (String) o.get("caption");
            // String sentence = (String) o.get("raw");
            // int id = Integer.parseInt(o.get("id").toString());
            int id = Integer.parseInt(o.get("image_id").toString());

            SceneGraph sg = parser.parse(sentence);

            String g = sg.toJSON(id, null, null);
            lines.add(g);

            if (i % 500 == 0) {
                System.out.println("thread " + index + ": " + i + " "+(System.currentTimeMillis() - start) / 1000.0);
                try {
                    File p = Paths.get(outFolder, Integer.toString(index)).toFile();
                    if (!p.exists()) {
                        p.mkdirs();
                    }
                    String outFile = Paths.get(p.getAbsolutePath(), i + ".list").toString();
                    writeFile(outFile, lines);
                } catch (Exception e) {
                    e.printStackTrace();
                    break;
                }
                lines.clear();
            }
        }
    }
}

public class Extract {

    static void writeFile(String file, ArrayList<String> lines) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter(file, "utf-8");
        for (String line : lines){
            writer.println(line);
        }
        writer.close();
    }

    public static void main(String[] args) throws Exception {
        String inFile = args[0];
        String outFolder = args[1];
        int numThread = Integer.parseInt(args[2]);

        // coco
        BufferedReader reader = new BufferedReader(new FileReader(inFile));
        JSONParser p = new JSONParser();
        JSONObject object = (JSONObject) p.parse(reader);
        JSONArray a = (JSONArray) object.get("annotations");
        // JSONArray a = (JSONArray) object.get("images");
        
        int chunkSize = (int) Math.ceil(((float)a.size()) / numThread);

        ArrayList<Worker> workers = new ArrayList<>();
        int i, j, ci = 0;
        for (i = 0; i < a.size(); ) {
            ArrayList<JSONObject> chunk = new ArrayList<>();

            for (j = 0; j < chunkSize && i < a.size(); j++, i++) {
                chunk.add((JSONObject) a.get(i));
            }

            System.out.println("chunk size: " + chunk.size());

            Worker w = new Worker(chunk, ci, outFolder);
            w.start();
            workers.add(w);
            ci += 1;
        }

        for (Worker w : workers) {
            w.join();
        }

    }
}
