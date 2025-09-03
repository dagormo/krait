using System.IO;

namespace Krait.Utils
{
    public static class CommonUtil
    {
        public static string Sanitize(string name)
        {
            if (string.IsNullOrEmpty(name)) return "Item";
            foreach (var c in Path.GetInvalidFileNameChars()) name = name.Replace(c, '_');
            return name;
        }
    }
}
