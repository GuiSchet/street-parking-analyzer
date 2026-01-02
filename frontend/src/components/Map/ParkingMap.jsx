import { Stage, Layer } from 'react-konva';
import ParkingSpace from './ParkingSpace';

const ParkingMap = ({ spaces, width = 800, height = 600 }) => {
  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700">
      <Stage width={width} height={height}>
        <Layer>
          {/* Dibujar espacios */}
          {spaces.map(space => (
            <ParkingSpace
              key={space.space_id}
              space={space}
            />
          ))}
        </Layer>
      </Stage>
    </div>
  );
};

export default ParkingMap;
