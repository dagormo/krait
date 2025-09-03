using System;
using Thermo.Chromeleon.Sdk.Interfaces;
using Thermo.Chromeleon.Sdk.Interfaces.Data;

namespace Krait.Chromeleon
{
    public static class InjectionHelpers
    {
        public static IInstrumentMethod ResolveInstrumentMethod(IInjection inj, IItemFactory fac)
        {
            if (inj == null) return null;

            try
            {
                var progProp = inj.GetType().GetProperty("Program") ??
                               inj.GetType().GetProperty("InstrumentMethod");

                if (progProp == null) return null;

                var val = progProp.GetValue(inj, null);

                if (val is IInstrumentMethod m) return m;

                if (val is Uri uriVal)
                {
                    if (fac.TryGetItem(uriVal, out IDataItem methodItem) && methodItem is IInstrumentMethod m2)
                        return m2;
                }
                else if (val is string strVal && Uri.TryCreate(strVal, UriKind.RelativeOrAbsolute, out var uri2))
                {
                    if (fac.TryGetItem(uri2, out IDataItem methodItem2) && methodItem2 is IInstrumentMethod m3)
                        return m3;
                }
            }
            catch { }

            return null;
        }
    }
}
