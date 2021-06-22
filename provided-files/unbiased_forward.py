import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from argparse import *
import PyFrensie.Geometry.DagMC as DagMC
import PyFrensie.Utility.Distribution as Distribution
import PyFrensie.MonteCarlo as MonteCarlo
import PyFrensie.MonteCarlo.Collision as Collision
import PyFrensie.MonteCarlo.ActiveRegion as ActiveRegion
import PyFrensie.MonteCarlo.Event as Event
import PyFrensie.MonteCarlo.Manager as Manager
import PyFrensie.Data as Data
import PyFrensie.Utility.Mesh as Mesh
import PyFrensie.Utility as Utility
import PyFrensie.Utility.MPI as MPI


if __name__ == "__main__":
  #--------------------------------------------------------------------------------#
  # SIMULATION PARAMETERS
  #--------------------------------------------------------------------------------#
  #GEOMETRY AND MATERIAL DENSITIES ARE IN THE .trelis FILE
  parser = ArgumentParser()
  parser.add_argument("--threads", dest="threads", type=int, default=1)
  parser.add_argument("--num_particles", dest="num_particles", type=float, default=1e2)
  args=parser.parse_args()

  ## Initialize the MPI session
  session = MPI.GlobalMPISession( len(sys.argv), sys.argv )

  # Suppress logging on all procs except for the master (proc=0)
  Utility.removeAllLogs()
  session.initializeLogs( 0, True )

  db_path = os.environ.get("DATABASE_PATH")
  sim_name = "forward"
  num_particles = args.num_particles
  threads = args.threads
  print(threads)
  if db_path is None:
      print('The database path must be specified!')
      sys.exit(1)
  #threshold_weight = 1e-3
  #survival_weight = 1
  ## Set the simulation properties for forward
  simulation_properties = MonteCarlo.SimulationProperties()
  
  #USE PHOTONS

  simulation_properties.setParticleMode( MonteCarlo.PHOTON_MODE )
  simulation_properties.setNumberOfHistories( num_particles )
  simulation_properties.setMaxRendezvousBatchSize( int(num_particles) )
  #simulation_properties.setPhotonRouletteThresholdWeight( threshold_weight )
  #simulation_properties.setPhotonRouletteSurvivalWeight( survival_weight )
  #simulation_properties.setImplicitCaptureModeOn()
  simulation_properties.setIncoherentModelType(MonteCarlo.FULL_PROFILE_DB_IMPULSE_INCOHERENT_MODEL)

  model_properties = DagMC.DagMCModelProperties("model.h5m")
  model_properties.setMaterialPropertyName( "material" )
  model_properties.setDensityPropertyName( "density" )
  model_properties.setTerminationCellPropertyName( "termination.cell" )
  model_properties.useFastIdLookup()
  model = DagMC.DagMCModel( model_properties )


  data_file_type = Data.PhotoatomicDataProperties.Native_EPR_FILE

  ## Set up the materials
  database = Data.ScatteringCenterPropertiesDatabase( db_path )
  
  #MATERIAL INFO. MATERIAL IS ASSIGNED IN TRELIS FILE ALONG WITH DENSITY. ZAID IS BELOW.

  # Extract the properties for Mn from the database
  Mn_properties = database.getAtomProperties( Data.ZAID(25000) )
  
  # Extract the properties for Ge from the database
  Ge_properties = database.getAtomProperties( Data.ZAID(32000) )

  # Extract the properties for H from the database
  H_properties = database.getAtomProperties( Data.ZAID(1000) )
  
  # Extract the properties for Pb from the database
  Pb_properties = database.getAtomProperties( Data.ZAID(82000) )

  # Set the definition for H, Pb, Mn, Ge for this simulation
  scattering_center_definitions = Collision.ScatteringCenterDefinitionDatabase()
  Mn_definition = scattering_center_definitions.createDefinition( "Mn", Data.ZAID(25000) )
  Ge_definition = scattering_center_definitions.createDefinition( "Ge", Data.ZAID(32000) )
  H_definition = scattering_center_definitions.createDefinition( "H", Data.ZAID(1000) )
  Pb_definition = scattering_center_definitions.createDefinition( "Pb", Data.ZAID(82000) )

  file_version = 0
  
  Mn_definition.setPhotoatomicDataProperties( Mn_properties.getSharedPhotoatomicDataProperties( data_file_type, file_version) )
  Ge_definition.setPhotoatomicDataProperties( Ge_properties.getSharedPhotoatomicDataProperties( data_file_type, file_version) )
  H_definition.setPhotoatomicDataProperties( H_properties.getSharedPhotoatomicDataProperties( data_file_type, file_version) )
  Pb_definition.setPhotoatomicDataProperties( Pb_properties.getSharedPhotoatomicDataProperties( data_file_type, file_version) )

  # Set the definition for materials
  material_definitions = Collision.MaterialDefinitionDatabase()

  material_definitions.addDefinition( "Mn", 1, ["Mn"], [1.0] )
  material_definitions.addDefinition( "Ge", 2, ["Ge"], [1.0] )
  material_definitions.addDefinition( "H", 3, ["H"], [1.0] )
  material_definitions.addDefinition( "Pb", 4, ["Pb"], [1.0] )

  filled_model = Collision.FilledGeometryModel( db_path, scattering_center_definitions, material_definitions, simulation_properties, model, True )

  #MESH INFORMATION. MESH IS 50x50x50 WITH EACH ELEMENT SIZE 2x2x2.

  xyz0=-25

  geometry_dimension_size = 50

  mesh_element_size = 2

  x_planes = []
  y_planes = []
  z_planes = []
  for i in range((geometry_dimension_size/mesh_element_size)+1):
    x_planes.append(xyz0 + i*mesh_element_size)
    y_planes.append(xyz0 + i*mesh_element_size)
    z_planes.append(xyz0 + i*mesh_element_size)

  estimator_mesh = Mesh.StructuredHexMesh(x_planes, y_planes, z_planes)

  source_centroid = [ -16.0, 0.0, 0.0 ]
  detector_centroid = [ 16.0, 0.0, 0.0 ]

  source_element = estimator_mesh.whichElementIsPointIn(source_centroid)
  detector_element = estimator_mesh.whichElementIsPointIn(detector_centroid)

  #SOURCE DESCRIPTION. SOURCE IS CONTAINED IN VOLUME 4

  source_x_raw_distribution = Distribution.UniformDistribution(-17, -15, 1)
  source_y_raw_distribution = Distribution.UniformDistribution(-1, 1, 1)
  source_z_raw_distribution = Distribution.UniformDistribution(-1, 1, 1)

  source_x_distribution = ActiveRegion.IndependentPrimarySpatialDimensionDistribution(source_x_raw_distribution)
  source_y_distribution = ActiveRegion.IndependentSecondarySpatialDimensionDistribution(source_y_raw_distribution)
  source_z_distribution = ActiveRegion.IndependentTertiarySpatialDimensionDistribution(source_z_raw_distribution)

  particle_distribution = ActiveRegion.StandardParticleDistribution("Forward source")
  particle_distribution.setDimensionDistribution(source_x_distribution)
  particle_distribution.setDimensionDistribution(source_y_distribution)
  particle_distribution.setDimensionDistribution(source_z_distribution)

  #MONOENERGETIC SOURCE AT 0.835 MeV

  particle_distribution.setEnergy(0.835)
  particle_distribution.constructDimensionDistributionDependencyTree()

  source_component = ActiveRegion.StandardPhotonSourceComponent(1, 1.0, model, particle_distribution)
  source = ActiveRegion.StandardParticleSource([source_component])

  event_handler = Event.EventHandler( model, simulation_properties )

  #I DON'T THINK MCNP HAS THIS KIND OF ESTIMATOR, SO IGNORE IF IT DOESN'T EXIST

  cell_integral_estimator = Event.WeightMultipliedCellCollisionFluxEstimator(1, 1.0, [5], model)
  cell_integral_estimator.setParticleTypes([MonteCarlo.PHOTON])
  cell_integral_estimator.setEnergyDiscretization([1e-3, 0.835])
  event_handler.addEstimator(cell_integral_estimator)

  #ESTIMATOR OF INTEREST - TRACK LENGTH ESTIMATOR IN VOLUME 5

  cell_integral_tl_estimator = Event.WeightMultipliedCellTrackLengthFluxEstimator(2, 1.0, [5], model)
  cell_integral_tl_estimator.setParticleTypes([MonteCarlo.PHOTON])
  cell_integral_tl_estimator.setEnergyDiscretization([1e-3, 0.835])
  event_handler.addEstimator(cell_integral_tl_estimator)

  #IGNORE OTHER ESTIMATORS

  detector_group_0_estimator = Event.WeightMultipliedCellTrackLengthFluxEstimator(3, 1.0, [5], model)
  detector_group_0_estimator.setParticleTypes([MonteCarlo.PHOTON])
  detector_group_0_estimator.setDirectionDiscretization(Event.PQLA, 2, False)
  detector_group_0_estimator.setEnergyDiscretization([1e-3,0.4175, 0.835])
  event_handler.addEstimator(detector_group_0_estimator)

  mesh_estimator = Event.WeightMultipliedMeshTrackLengthFluxEstimator(4, 1.0, estimator_mesh)
  mesh_estimator.setParticleTypes([MonteCarlo.PHOTON])
  mesh_estimator.setDirectionDiscretization(Event.PQLA, 2, False)
  mesh_estimator.setEnergyDiscretization([1e-3,0.4175, 0.835])
  event_handler.addEstimator(mesh_estimator)


  factory = Manager.ParticleSimulationManagerFactory( filled_model,
                                                  source,
                                                  event_handler,
                                                  simulation_properties,
                                                  sim_name,
                                                  "xml",
                                                  threads )

  manager = factory.getManager()
  manager.useSingleRendezvousFile()

  session.restoreOutputStreams()

  ## Run the simulation
  manager.runSimulation()

  #collision_estimator_processed_data = manager.getEventHandler().getEstimator(1).getTotalProcessedData()
  #print("Collision estimator mean: ", collision_estimator_processed_data["mean"])
  #print("Collision estimator RE: ", collision_estimator_processed_data["re"])
  #print("Collision estimator FOM: ", collision_estimator_processed_data["fom"])
  #print("Collision estimator VoV: ", collision_estimator_processed_data["vov"])

  #collision_direction_estimator_processed_data = manager.getEventHandler().getEstimator(2).getTotalBinProcessedData()
  #print("Direction estimator means: ", collision_direction_estimator_processed_data["mean"])
  #print("Direction estimator RE: ", collision_direction_estimator_processed_data["re"])
  #print("Direction estimator FOM: ", collision_direction_estimator_processed_data["fom"])
  #print("Direction estimator VoV: ", collision_direction_estimator_processed_data["vov"])

  #mesh_detector_biasing_processed_data = manager.getEventHandler().getEstimator(3).getEntityBinProcessedData(detector_element)
  #print("Mesh detector estimator means: ", mesh_detector_biasing_processed_data["mean"])
  #print("Mesh detector estimator RE: ", mesh_detector_biasing_processed_data["re"])
  #print("Mesh detector estimator FOM: ", mesh_detector_biasing_processed_data["fom"])
  #print("Mesh detector estimator VoV: ", mesh_detector_biasing_processed_data["vov"])