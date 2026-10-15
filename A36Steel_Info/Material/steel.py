import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt



# USER INPUT
XML_FILE = "A36_datasheet.xml"



# ---------- Helper Functions --------------

#Convert comma-separated XML data into a NumPy float array.
def string_to_array(text):

    return np.array([float(value.strip()) for value in text.split(",")], dtype = float)


#Extract a temperature-dependent material property from the ANSYS / Granta EngineeringData XML structure.
def extract_temperature_property(root, property_name, parameter_name):
  
    # Search through all PropertyData elements
    for property_data in root.iter("PropertyData"):

        if property_data.get("property") == property_name:

            temperature = None
            values = None

            # Search through ParameterValue entries
            for parameter in property_data.findall("ParameterValue"):

                parameter_id = parameter.get("parameter")

                data_element = parameter.find("Data")

                if data_element is None or data_element.text is None:
                    continue

                # Property values
                if parameter_id == parameter_name:
                    values = string_to_array(data_element.text)

                # Corresponding temperatures
                elif parameter_id == "Temperature":
                    temperature = string_to_array(data_element.text)

            if temperature is not None and values is not None:

                if len(temperature) != len(values):
                    raise ValueError(
                        f"{property_name}: number of temperatures "
                        f"({len(temperature)}) does not match number "
                        f"of property values ({len(values)})."
                    )

                return temperature, values

    raise ValueError(
        f'Could not find property "{property_name}" in XML file.'
    )



#Read XML File
tree = ET.parse(XML_FILE)
root = tree.getroot()
material_name = root.findtext( ".//Material/BulkDetails/Name", default="Material")
print(f"\nMaterial: {material_name}")


#Exact Thermal Conductivity 
T_k, k = extract_temperature_property( root, property_name="Thermal Conductivity", parameter_name="Thermal Conductivity")
T_cp, cp = extract_temperature_property(root, property_name="Specific Heat", parameter_name="Specific Heat")


#Print Data

print("--------------------------------------------")
print("\nThermal Conductivity")
print("--------------------------------------------")
print("Temperature [K]     k [W/(m*K)]")

for T, value in zip(T_k, k):
    print(f"{T:10.2f}          {value:10.3f}")


print("--------------------------------------------")
print("\nSpecific Heat")
print("--------------------------------------------")
print("Temperature [K]     Cp [J/(kg*K)]")

for T, value in zip(T_cp, cp):
    print(f"{T:10.2f}          {value:10.3f}")

#Plot
plt.figure(figsize=(8, 6))

plt.plot(
    T_k,
    k,
    marker="o",
    linewidth=2
)

plt.xlabel("Temperature [K]")
plt.ylabel("Thermal Conductivity, k [W/(m·K)]")
plt.title(
    f"Thermal Conductivity vs Temperature\n{material_name}"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT SPECIFIC HEAT
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(T_cp, cp, marker="o", linewidth=2)

plt.xlabel("Temperature [K]")
plt.ylabel("Specific Heat Capacity, $c_p$ [J/(kg·K)]")
plt.title(f"Specific Heat Capacity vs Temperature\n{material_name}")

plt.grid(True)
plt.tight_layout()
plt.show()